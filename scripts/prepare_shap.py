"""SHAP 기여도 캐시 생성 (한 번 실행).

fold0 LGB + fold0 CB TreeExplainer로 test 전체(50000행) SHAP 계산 후
가중 합산해서 data/test_shap.npy 에 저장.

소요시간: ~20초 (LGB 17s + CB 2s).
산출물:
    data/test_shap.npy   — (50000, 168) float32, w_lgb*shap_lgb + w_cb*shap_cb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_DIR, ROOT

OUT = ROOT / "data" / "test_shap.npy"
TEST_FEATURES = ROOT / "data" / "test_features.parquet"


def main() -> None:
    if not TEST_FEATURES.exists():
        raise FileNotFoundError(f"피처 캐시 없음. 먼저 prepare_features.py 실행.\n  expected: {TEST_FEATURES}")

    weights_path = MODEL_DIR / "weights.json"
    raw_w = json.loads(weights_path.read_text())
    w_lgb = float(raw_w.get("lgb", 0.0))
    w_cb  = float(raw_w.get("cb",  0.0))
    total = w_lgb + w_cb
    w_lgb_n, w_cb_n = w_lgb / total, w_cb / total

    import lightgbm as lgb
    import catboost as cb

    print("loading models ...")
    m_lgb = lgb.Booster(model_file=str(MODEL_DIR / "lgb_fold0.txt"))
    m_cb  = cb.CatBoostRegressor()
    m_cb.load_model(str(MODEL_DIR / "cb_fold0.cbm"))
    feat_cols = m_lgb.feature_name()
    print(f"  feature cols: {len(feat_cols)}")

    print("loading test features ...")
    feats = pd.read_parquet(TEST_FEATURES)
    X = feats[feat_cols].values.astype(np.float32)
    print(f"  X shape: {X.shape}")

    print("computing LGB SHAP ...")
    ex_lgb = shap.TreeExplainer(m_lgb)
    sv_lgb = ex_lgb.shap_values(X).astype(np.float32)
    print(f"  lgb shap shape: {sv_lgb.shape}")

    print("computing CB SHAP ...")
    ex_cb = shap.TreeExplainer(m_cb)
    sv_cb = ex_cb.shap_values(X).astype(np.float32)
    print(f"  cb  shap shape: {sv_cb.shape}")

    combined = (w_lgb_n * sv_lgb + w_cb_n * sv_cb).astype(np.float32)
    OUT.parent.mkdir(exist_ok=True)
    np.save(OUT, combined)

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"\nsaved → {OUT}  shape {combined.shape}  ({size_mb:.1f} MB)")
    print(f"weights applied: w_lgb_n={w_lgb_n:.4f}  w_cb_n={w_cb_n:.4f}")

    top3 = sorted(zip(feat_cols, combined[0]), key=lambda t: abs(t[1]), reverse=True)[:3]
    print(f"sample[0] top3: {[(f, round(v, 4)) for f, v in top3]}")


if __name__ == "__main__":
    main()
