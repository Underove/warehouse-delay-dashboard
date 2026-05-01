"""
앙상블 추론 모듈.

`models/weights.json` 이 있고 fold 모델 파일이 갖춰지면 EnsemblePredictor가 동작하고,
없으면 MockPredictor가 더미 예측을 돌려준다 (UI 개발용).

현재 활성: LGB + CB tree-only (Phase A). GRU/Attn은 다음 단계에서 추가.

학습 스크립트는 log1p 변환 + GroupKFold(by layout_id) 5-fold. 추론은:
    pred = expm1( mean_over_folds( log_pred ) )  per model
    final = sum_models( w_normalized * pred )
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_DIR, RISK_THRESHOLDS, ROOT, TARGET

TEST_FEATURES_PARQUET = ROOT / "data" / "test_features.parquet"
N_FOLDS = 5


@dataclass
class Prediction:
    value: float
    risk: str
    contributions: dict[str, float] | None = None


def _classify_risk(minutes: float) -> str:
    if minutes >= RISK_THRESHOLDS["critical"]:
        return "critical"
    if minutes >= RISK_THRESHOLDS["warning"]:
        return "warning"
    return "normal"


class MockPredictor:
    """모델 파일 도착 전까지 UI 개발용. row의 핵심 신호로 직관적인 더미 예측."""

    SIGNAL_COLS = (
        "order_inflow_15m",
        "congestion_score",
        "robot_active",
        "pack_utilization",
    )

    def __init__(self, ref_df: pd.DataFrame | None = None) -> None:
        self.ref_stats: dict[str, tuple[float, float]] = {}
        if ref_df is not None:
            for c in self.SIGNAL_COLS:
                if c in ref_df.columns:
                    self.ref_stats[c] = (
                        float(ref_df[c].mean(skipna=True)),
                        float(ref_df[c].std(skipna=True) or 1.0),
                    )

    def _signal_contributions(self, row: pd.Series) -> dict[str, float]:
        contribs: dict[str, float] = {}
        for c in self.SIGNAL_COLS:
            if c not in self.ref_stats or c not in row or pd.isna(row[c]):
                continue
            mean, std = self.ref_stats[c]
            z = (float(row[c]) - mean) / std
            contribs[c] = round(z * 1.5, 3)
        return contribs

    def predict_one(self, row: pd.Series) -> Prediction:
        base = 18.96
        contribs = self._signal_contributions(row)
        z_total = sum(v for v in contribs.values()) / 1.5
        value = max(0.0, base + 1.5 * z_total)
        return Prediction(value=value, risk=_classify_risk(value), contributions=contribs)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(r).value for _, r in df.iterrows()])


class EnsemblePredictor(MockPredictor):
    """
    LGB + CB 5-fold 앙상블 (Tree-only Phase A).

    - 입력 row 의 `ID` 로 캐시된 test_features.parquet 에서 피처 조회
    - 각 모델: log1p 학습 → expm1 복원, 5-fold 평균
    - weights.json 의 lgb/cb 가중치를 둘만 합 1로 재정규화
    - GRU/Attn 가중치는 일단 무시 (xgb=0 동일 처리)
    - contributions 는 MockPredictor 의 z-score 방식 유지 (SHAP 통합 전)
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        ref_df: pd.DataFrame | None = None,
        features_parquet: Path = TEST_FEATURES_PARQUET,
    ) -> None:
        super().__init__(ref_df=ref_df)

        if not features_parquet.exists():
            raise FileNotFoundError(
                f"피처 캐시 없음. 먼저 `python scripts/prepare_features.py` 실행.\n"
                f"  expected: {features_parquet}"
            )

        weights_path = model_dir / "weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"weights.json 없음: {weights_path}")
        raw_w = json.loads(weights_path.read_text())

        w_lgb = float(raw_w.get("lgb", 0.0))
        w_cb = float(raw_w.get("cb", 0.0))
        total = w_lgb + w_cb
        if total <= 0:
            raise ValueError("lgb + cb 가중치가 0 이하 — Tree-only 추론 불가")
        self.w_lgb = w_lgb / total
        self.w_cb = w_cb / total
        self.active_models = ("lgb", "cb")

        import lightgbm as lgb
        import catboost as cb

        self.lgb_models = [
            lgb.Booster(model_file=str(model_dir / f"lgb_fold{i}.txt")) for i in range(N_FOLDS)
        ]
        self.cb_models = []
        for i in range(N_FOLDS):
            m = cb.CatBoostRegressor()
            m.load_model(str(model_dir / f"cb_fold{i}.cbm"))
            self.cb_models.append(m)

        self.feature_cols = list(self.lgb_models[0].feature_name())

        feats = pd.read_parquet(features_parquet)
        if "ID" not in feats.columns:
            raise ValueError("test_features.parquet 에 ID 컬럼 없음")
        self.feature_table = feats.set_index("ID", drop=False)

    def _predict_features(self, X: pd.DataFrame) -> np.ndarray:
        """피처 DataFrame → 최종 예측 분(min) 배열."""
        X = X[self.feature_cols]

        lgb_log = np.zeros(len(X))
        for m in self.lgb_models:
            lgb_log += m.predict(X.values) / N_FOLDS
        lgb_pred = np.expm1(lgb_log)

        cb_log = np.zeros(len(X))
        for m in self.cb_models:
            cb_log += m.predict(X.values) / N_FOLDS
        cb_pred = np.expm1(cb_log)

        final = self.w_lgb * lgb_pred + self.w_cb * cb_pred
        return np.clip(final, 0.0, None)

    def predict_one(self, row: pd.Series) -> Prediction:
        rid = row.get("ID")
        contribs = self._signal_contributions(row)
        if rid is None or rid not in self.feature_table.index:
            return super().predict_one(row)

        feat_row = self.feature_table.loc[[rid]]
        # row 의 SIGNAL_COLS 가 캐시 baseline 과 다르면 What-If 모드 — 단일 row 재계산
        overrides: dict[str, float] = {}
        for c in self.SIGNAL_COLS:
            if c in row.index and c in feat_row.columns and not pd.isna(row[c]):
                cached = feat_row[c].iloc[0]
                if pd.isna(cached) or abs(float(row[c]) - float(cached)) > 1e-9:
                    overrides[c] = float(row[c])

        if overrides:
            from src.preprocess import recompute_signal_dependent
            feat_row = feat_row.copy()
            for c, v in overrides.items():
                feat_row[c] = v
            feat_row = recompute_signal_dependent(feat_row)

        value = float(self._predict_features(feat_row)[0])
        return Prediction(value=value, risk=_classify_risk(value), contributions=contribs)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "ID" not in df.columns:
            return super().predict(df)
        ids = [i for i in df["ID"] if i in self.feature_table.index]
        if not ids:
            return super().predict(df)
        feat_rows = self.feature_table.loc[ids]
        preds = self._predict_features(feat_rows)
        return preds


def load_predictor(ref_df: pd.DataFrame | None = None):
    """모델 파일이 있으면 EnsemblePredictor, 없으면 MockPredictor."""
    if (MODEL_DIR / "weights.json").exists() and TEST_FEATURES_PARQUET.exists():
        return EnsemblePredictor(ref_df=ref_df)
    return MockPredictor(ref_df=ref_df)


if __name__ == "__main__":
    from data_loader import load_test, load_train

    train = load_train()
    test = load_test()
    predictor = load_predictor(ref_df=train)
    print(f"predictor: {type(predictor).__name__}")
    if isinstance(predictor, EnsemblePredictor):
        print(f"  active models: {predictor.active_models}")
        print(f"  weights: lgb={predictor.w_lgb:.4f}  cb={predictor.w_cb:.4f}")
        print(f"  feature cols: {len(predictor.feature_cols)}")

    sample = test.iloc[0]
    pred = predictor.predict_one(sample)
    print(f"sample[0]: pred={pred.value:.2f} min  risk={pred.risk}  ID={sample.get('ID')}")

    sample2 = test.iloc[12345]
    pred2 = predictor.predict_one(sample2)
    print(f"sample[12345]: pred={pred2.value:.2f} min  risk={pred2.risk}  ID={sample2.get('ID')}")

    batch = predictor.predict(test.head(100))
    print(f"batch[100]: mean={batch.mean():.2f}  min={batch.min():.2f}  max={batch.max():.2f}")
