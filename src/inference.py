"""
앙상블 추론 모듈 (Phase B — 4-model full ensemble).

EnsemblePredictor: LGB + CB + BiGRU + BiGRU+Attn.
- Tree : ID → test_features.parquet lookup, What-If signal recompute 지원
- Seq  : ID → test_seq_index.parquet로 (scenario_idx, ts_rank) 조회
         → test_seq.npy[scenario_idx] 5-fold forward, log1p mean → expm1
         What-If에는 비반응 (단일 row 한계 — baseline 시퀀스 그대로 사용)
- 가중치: weights.json 그대로 (lgb+cb+gru+attn 합 ≈ 1.0, 재정규화 없음)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_DIR, RISK_THRESHOLDS, ROOT

TEST_FEATURES_PARQUET = ROOT / "data" / "test_features.parquet"
TEST_SEQ_NPY          = ROOT / "data" / "test_seq.npy"
TEST_SEQ_INDEX        = ROOT / "data" / "test_seq_index.parquet"
TEST_SHAP_NPY         = ROOT / "data" / "test_shap.npy"
SEQ_SCALER_PKL        = MODEL_DIR / "seq_scaler.pkl"
N_FOLDS    = 5
TOP_SHAP   = 8     # 기여도 차트에 표시할 피처 수
DEVICE     = torch.device("cpu")


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


# ── 시퀀스 모델 아키텍처 ─────────────────────────────────────────────────────

class _BiGRU(nn.Module):
    """Bi-GRU 회귀 헤드. 학습 파라미터: hidden=128, layers=2, dropout=0.3."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(self.dropout(out)).squeeze(-1)  # (batch, 25)


class _BiGRUAttn(nn.Module):
    """Bi-GRU + Multi-Head Self-Attention. 학습 파라미터: hidden=128, layers=2, dropout=0.3, heads=4."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_heads: int = 4) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm    = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        out = self.norm(gru_out + attn_out)
        return self.head(self.dropout(out)).squeeze(-1)  # (batch, 25)


# ── MockPredictor ─────────────────────────────────────────────────────────────

class MockPredictor:
    """모델 파일 없을 때 UI 개발용. row의 핵심 신호 z-score 기반 더미 예측."""

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


# ── EnsemblePredictor ─────────────────────────────────────────────────────────

class EnsemblePredictor(MockPredictor):
    """
    LGB + CB + BiGRU + BiGRU+Attn 4-model 앙상블.

    시퀀스 모델은 시나리오 단위로 한 번만 forward하고 내부 캐시(_seq_cache)에 저장.
    기여도: test_shap.npy 캐시 O(1) lookup (What-If 시 단일 row 즉시 계산).
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        ref_df: pd.DataFrame | None = None,
        features_parquet: Path = TEST_FEATURES_PARQUET,
    ) -> None:
        super().__init__(ref_df=ref_df)

        # ── 가중치 ──────────────────────────────────────────────────────────
        weights_path = model_dir / "weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"weights.json 없음: {weights_path}")
        raw_w = json.loads(weights_path.read_text())
        self.w_lgb  = float(raw_w.get("lgb",  0.0))
        self.w_cb   = float(raw_w.get("cb",   0.0))
        self.w_gru  = float(raw_w.get("gru",  0.0))
        self.w_attn = float(raw_w.get("attn", 0.0))
        self.active_models = ("lgb", "cb", "gru", "attn")

        # ── Tree 모델 ────────────────────────────────────────────────────────
        if not features_parquet.exists():
            raise FileNotFoundError(
                f"피처 캐시 없음. 먼저 `python scripts/prepare_features.py` 실행.\n"
                f"  expected: {features_parquet}"
            )
        import lightgbm as lgb
        import catboost as cb

        self.lgb_models = [
            lgb.Booster(model_file=str(model_dir / f"lgb_fold{i}.txt"))
            for i in range(N_FOLDS)
        ]
        self.cb_models = []
        for i in range(N_FOLDS):
            m = cb.CatBoostRegressor()
            m.load_model(str(model_dir / f"cb_fold{i}.cbm"))
            self.cb_models.append(m)

        self.tree_feature_cols = list(self.lgb_models[0].feature_name())

        feats = pd.read_parquet(features_parquet)
        if "ID" not in feats.columns:
            raise ValueError("test_features.parquet 에 ID 컬럼 없음")
        self.feature_table = feats.set_index("ID", drop=False)

        # ── 시퀀스 모델 ──────────────────────────────────────────────────────
        seq_ok = (
            TEST_SEQ_NPY.exists()
            and TEST_SEQ_INDEX.exists()
            and SEQ_SCALER_PKL.exists()
            and all((model_dir / f"gru_fold{i}.pt").exists() for i in range(N_FOLDS))
            and all((model_dir / f"attn_fold{i}.pt").exists() for i in range(N_FOLDS))
        )
        self._seq_enabled = seq_ok

        if seq_ok:
            self._test_seq = np.load(TEST_SEQ_NPY)       # (2000, 25, 173)
            seq_index = pd.read_parquet(TEST_SEQ_INDEX)
            self._seq_index = seq_index.set_index("ID")  # ID → scenario_idx, ts_rank

            scaler_bundle = joblib.load(SEQ_SCALER_PKL)
            n_features: int = scaler_bundle["n_features"]

            self._gru_models = [
                self._load_pt(model_dir / f"gru_fold{i}.pt",
                              _BiGRU(input_size=n_features))
                for i in range(N_FOLDS)
            ]
            self._attn_models = [
                self._load_pt(model_dir / f"attn_fold{i}.pt",
                              _BiGRUAttn(input_size=n_features))
                for i in range(N_FOLDS)
            ]
            self._seq_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            print(f"[inference] seq models loaded  shape={self._test_seq.shape}")
        else:
            print("[inference] seq models unavailable — running tree-only")
            total = self.w_lgb + self.w_cb
            if total > 0:
                self.w_lgb /= total
                self.w_cb  /= total
            self.active_models = ("lgb", "cb")

        # ── SHAP 캐시 ────────────────────────────────────────────────────────
        if TEST_SHAP_NPY.exists():
            import shap as _shap
            self._shap_arr = np.load(TEST_SHAP_NPY)           # (50000, 168)
            self._id_to_shap_idx = {
                rid: i for i, rid in enumerate(feats["ID"])
            }
            # What-If 단일 row 계산용 fold0 explainer
            self._shap_ex_lgb = _shap.TreeExplainer(self.lgb_models[0])
            self._shap_ex_cb  = _shap.TreeExplainer(self.cb_models[0])
            w_total = self.w_lgb + self.w_cb
            self._shap_w_lgb = self.w_lgb / w_total if w_total > 0 else 0.5
            self._shap_w_cb  = self.w_cb  / w_total if w_total > 0 else 0.5
            self._shap_enabled = True
            print(f"[inference] shap cache loaded  shape={self._shap_arr.shape}")
        else:
            self._shap_enabled = False
            print("[inference] shap cache unavailable — using z-score contributions")

    @staticmethod
    def _load_pt(path: Path, model: nn.Module) -> nn.Module:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model

    # ── 내부 추론 헬퍼 ────────────────────────────────────────────────────────

    def _tree_preds(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """→ (lgb_pred_min, cb_pred_min) 각각 expm1 복원."""
        Xv = X[self.tree_feature_cols].values

        lgb_log = np.zeros(len(X))
        for m in self.lgb_models:
            lgb_log += m.predict(Xv) / N_FOLDS

        cb_log = np.zeros(len(X))
        for m in self.cb_models:
            cb_log += m.predict(Xv) / N_FOLDS

        return np.expm1(lgb_log), np.expm1(cb_log)

    def _seq_preds(self, scenario_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """시나리오 전체 25-step GRU/Attn 예측. 결과를 캐시."""
        if scenario_idx in self._seq_cache:
            return self._seq_cache[scenario_idx]

        x = torch.tensor(
            self._test_seq[scenario_idx][np.newaxis],  # (1, 25, 173)
            dtype=torch.float32, device=DEVICE,
        )
        with torch.no_grad():
            gru_log = sum(m(x).cpu().numpy() for m in self._gru_models) / N_FOLDS
            attn_log = sum(m(x).cpu().numpy() for m in self._attn_models) / N_FOLDS

        gru_preds  = np.expm1(gru_log[0])   # (25,)
        attn_preds = np.expm1(attn_log[0])  # (25,)

        self._seq_cache[scenario_idx] = (gru_preds, attn_preds)
        return gru_preds, attn_preds

    def _shap_contributions(
        self,
        feat_row: pd.DataFrame,
        pred_minutes: float,
        rid: str | None,
        has_overrides: bool,
    ) -> dict[str, float]:
        """SHAP 기여도 → top 8 피처 dict (값: 근사 분 단위).

        캐시 hit: O(1) lookup.
        What-If (has_overrides=True): fold0 LGB+CB 단일 row 즉시 계산.
        scale: shap_i_log * (pred_minutes + 1)  — log1p 공간 Jacobian 근사.
        """
        if not self._shap_enabled:
            return self._signal_contributions(feat_row.iloc[0])

        X = feat_row[self.tree_feature_cols].values  # (1, 168)

        if has_overrides or rid is None or rid not in self._id_to_shap_idx:
            sv_lgb = self._shap_ex_lgb.shap_values(X)[0]
            sv_cb  = self._shap_ex_cb.shap_values(X)[0]
            sv = self._shap_w_lgb * sv_lgb + self._shap_w_cb * sv_cb
        else:
            sv = self._shap_arr[self._id_to_shap_idx[rid]]  # (168,)

        scale = pred_minutes + 1.0
        contribs_raw = sv * scale

        top_idx = np.argsort(np.abs(contribs_raw))[-TOP_SHAP:]
        return {
            self.tree_feature_cols[i]: round(float(contribs_raw[i]), 3)
            for i in top_idx
        }

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def predict_one(self, row: pd.Series) -> Prediction:
        rid = row.get("ID")

        if rid is None or rid not in self.feature_table.index:
            return super().predict_one(row)

        # Tree 예측 (What-If signal override 반영)
        feat_row = self.feature_table.loc[[rid]]
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

        lgb_pred, cb_pred = self._tree_preds(feat_row)
        lgb_val = float(lgb_pred[0])
        cb_val  = float(cb_pred[0])

        # 시퀀스 예측 (baseline 시퀀스 — What-If 비반응)
        if self._seq_enabled and rid in self._seq_index.index:
            row_meta = self._seq_index.loc[rid]
            scenario_idx = int(row_meta["scenario_idx"])
            ts_rank      = int(row_meta["ts_rank"])
            gru_preds, attn_preds = self._seq_preds(scenario_idx)
            gru_val  = float(np.clip(gru_preds[ts_rank],  0.0, None))
            attn_val = float(np.clip(attn_preds[ts_rank], 0.0, None))
            value = (
                self.w_lgb  * lgb_val
                + self.w_cb   * cb_val
                + self.w_gru  * gru_val
                + self.w_attn * attn_val
            )
        else:
            value = self.w_lgb * lgb_val + self.w_cb * cb_val

        value = max(0.0, value)
        contribs = self._shap_contributions(feat_row, value, rid, bool(overrides))
        return Prediction(value=value, risk=_classify_risk(value), contributions=contribs)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "ID" not in df.columns:
            return super().predict(df)
        return np.array([self.predict_one(r) .value for _, r in df.iterrows()])


# ── 팩토리 ────────────────────────────────────────────────────────────────────

def load_predictor(ref_df: pd.DataFrame | None = None) -> MockPredictor:
    """모델 파일이 갖춰지면 EnsemblePredictor, 없으면 MockPredictor."""
    if (MODEL_DIR / "weights.json").exists() and TEST_FEATURES_PARQUET.exists():
        return EnsemblePredictor(ref_df=ref_df)
    return MockPredictor(ref_df=ref_df)


if __name__ == "__main__":
    from src.data_loader import load_test, load_train

    train = load_train()
    test  = load_test()
    predictor = load_predictor(ref_df=train)
    print(f"predictor: {type(predictor).__name__}")
    if isinstance(predictor, EnsemblePredictor):
        print(f"  active models : {predictor.active_models}")
        print(f"  weights       : lgb={predictor.w_lgb:.4f}  cb={predictor.w_cb:.4f}"
              f"  gru={predictor.w_gru:.4f}  attn={predictor.w_attn:.4f}")
        print(f"  seq enabled   : {predictor._seq_enabled}")

    sample = test.iloc[0]
    pred = predictor.predict_one(sample)
    print(f"sample[0]     : pred={pred.value:.2f} min  risk={pred.risk}  ID={sample.get('ID')}")

    sample2 = test.iloc[12345]
    pred2 = predictor.predict_one(sample2)
    print(f"sample[12345] : pred={pred2.value:.2f} min  risk={pred2.risk}  ID={sample2.get('ID')}")

    batch = predictor.predict(test.head(50))
    print(f"batch[50]     : mean={batch.mean():.2f}  min={batch.min():.2f}  max={batch.max():.2f}")
