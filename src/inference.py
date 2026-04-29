"""
앙상블 추론 모듈.

학습된 모델 파일이 `models/`에 들어오면 실제 추론으로 동작하고,
없으면 MockPredictor가 합리적인 더미 예측을 돌려준다 (Phase 1 UI 개발용).

모델 파일 컨벤션 (예정):
    models/lgb_fold{0..4}.txt
    models/xgb_fold{0..4}.json
    models/cb_fold{0..4}.cbm
    models/gru_fold{0..4}.pt
    models/weights.json   # {"lgb": 0.32, "xgb": 0.24, "cb": 0.43, "gru": ...}
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_DIR, RISK_THRESHOLDS, TARGET


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
        self.ref_stats = {}
        if ref_df is not None:
            for c in self.SIGNAL_COLS:
                if c in ref_df.columns:
                    self.ref_stats[c] = (
                        float(ref_df[c].mean(skipna=True)),
                        float(ref_df[c].std(skipna=True) or 1.0),
                    )

    def predict_one(self, row: pd.Series) -> Prediction:
        base = 18.96  # train 평균
        z_total = 0.0
        contributions: dict[str, float] = {}
        for c in self.SIGNAL_COLS:
            if c not in self.ref_stats or c not in row or pd.isna(row[c]):
                continue
            mean, std = self.ref_stats[c]
            z = (float(row[c]) - mean) / std
            z_total += z
            contributions[c] = round(z * 1.5, 3)
        value = max(0.0, base + 1.5 * z_total)
        return Prediction(value=value, risk=_classify_risk(value), contributions=contributions)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(r).value for _, r in df.iterrows()])


class EnsemblePredictor:
    """실제 모델 5개(LGB/XGB/CB/GRU 등) 가중 평균. 미구현 — 모델 파일 도착 시 구현."""

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir = model_dir
        weights_path = model_dir / "weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"weights.json 없음. 모델 파일이 아직 없으면 MockPredictor를 쓰자.\n"
                f"  expected: {weights_path}"
            )
        self.weights = json.loads(weights_path.read_text())
        raise NotImplementedError("모델 로드 로직은 모델 파일 컨벤션 확정 후 구현")


def load_predictor(ref_df: pd.DataFrame | None = None):
    """모델 파일이 있으면 EnsemblePredictor, 없으면 MockPredictor."""
    if (MODEL_DIR / "weights.json").exists():
        return EnsemblePredictor()
    return MockPredictor(ref_df=ref_df)


if __name__ == "__main__":
    from data_loader import load_test, load_train

    train = load_train()
    test = load_test()
    predictor = load_predictor(ref_df=train)
    print(f"predictor: {type(predictor).__name__}")

    sample = test.iloc[0]
    pred = predictor.predict_one(sample)
    print(f"sample prediction: {pred.value:.2f} min  risk={pred.risk}")
    if pred.contributions:
        print("contributions (z*1.5):")
        for k, v in pred.contributions.items():
            print(f"  {k:25s}  {v:+.3f}")
