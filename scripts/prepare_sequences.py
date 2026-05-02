"""GRU/Attn 시퀀스 추론용 캐시 생성.

- train.csv 로 StandardScaler refit (노트북 7-2 시퀀스 학습과 동일한 fit)
- test 를 transform 해서 (n_test_scenarios, 25, N_FEATURES) 로 reshape
- ID → (scenario_idx, ts_rank) 매핑 parquet 도 함께 저장

산출물:
    models/seq_scaler.pkl
    data/test_seq.npy
    data/test_seq_index.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_DIR, ROOT, SEQ_LEN
from src.data_loader import load_layout, load_test, load_train
from src.preprocess import build_features, get_optimized_cols

SEQ_OUT = ROOT / "data" / "test_seq.npy"
INDEX_OUT = ROOT / "data" / "test_seq_index.parquet"
SCALER_OUT = MODEL_DIR / "seq_scaler.pkl"


def main() -> None:
    print("loading raw csvs ...")
    train_raw = load_train()
    test_raw = load_test()
    layout_raw = load_layout()
    print(f"  train {train_raw.shape}  test {test_raw.shape}  layout {layout_raw.shape}")

    print("running feature engineering ...")
    train_fe, test_fe = build_features(train_raw, test_raw, layout_raw)
    print(f"  train_fe {train_fe.shape}  test_fe {test_fe.shape}")

    opt_cols = get_optimized_cols(train_fe)
    n_features = len(opt_cols)
    print(f"  optimized cols: {n_features}")

    # build_features 내부에서 scenario_id, id_num 정렬됨 — 안전하게 한 번 더 확인
    train_fe = train_fe.sort_values(["scenario_id", "id_num"]).reset_index(drop=True)
    test_fe = test_fe.sort_values(["scenario_id", "id_num"]).reset_index(drop=True)

    if len(train_fe) % SEQ_LEN != 0:
        raise ValueError(f"train rows {len(train_fe)} 가 SEQ_LEN={SEQ_LEN} 의 배수가 아님")
    if len(test_fe) % SEQ_LEN != 0:
        raise ValueError(f"test rows {len(test_fe)} 가 SEQ_LEN={SEQ_LEN} 의 배수가 아님")

    print("fitting StandardScaler on train ...")
    scaler = StandardScaler()
    scaler.fit(train_fe[opt_cols].values)

    print("transforming test + reshaping ...")
    X_test_2d = scaler.transform(test_fe[opt_cols].values)
    n_test_scenarios = len(test_fe) // SEQ_LEN
    X_test_seq = X_test_2d.reshape(n_test_scenarios, SEQ_LEN, n_features).astype(np.float32)

    # ID → (scenario_idx, ts_rank) 매핑
    test_fe = test_fe.copy()
    test_fe["scenario_idx"] = np.repeat(np.arange(n_test_scenarios), SEQ_LEN)
    index_df = test_fe[["ID", "scenario_id", "scenario_idx", "ts_rank"]].copy()

    SEQ_OUT.parent.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    np.save(SEQ_OUT, X_test_seq)
    index_df.to_parquet(INDEX_OUT, index=False)
    joblib.dump(
        {"scaler": scaler, "feature_cols": opt_cols, "n_features": n_features},
        SCALER_OUT,
    )

    print(f"saved → {SEQ_OUT}  shape {X_test_seq.shape}  ({SEQ_OUT.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"saved → {INDEX_OUT}  rows {len(index_df)}")
    print(f"saved → {SCALER_OUT}  features {n_features}")

    print("\nsanity checks:")
    print(f"  scaler.mean_  range [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]")
    print(f"  scaler.scale_ range [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")
    print(f"  X_test_seq mean {X_test_seq.mean():.4f}  std {X_test_seq.std():.4f}")
    print(f"  first 5 IDs: {index_df['ID'].head().tolist()}")


if __name__ == "__main__":
    main()
