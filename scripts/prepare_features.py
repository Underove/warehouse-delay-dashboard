"""한 번 실행해서 test 피처를 캐시한다 (data/test_features.parquet)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT
from src.data_loader import load_layout, load_test, load_train
from src.preprocess import build_features, get_optimized_cols, get_reduced_cols

OUT = ROOT / "data" / "test_features.parquet"


def main() -> None:
    print("loading raw csvs ...")
    train_raw = load_train()
    test_raw = load_test()
    layout_raw = load_layout()
    print(f"  train {train_raw.shape}  test {test_raw.shape}  layout {layout_raw.shape}")

    print("running feature engineering ...")
    train_fe, test_fe = build_features(train_raw, test_raw, layout_raw)
    print(f"  train_fe {train_fe.shape}  test_fe {test_fe.shape}")

    opt = get_optimized_cols(train_fe)
    red = get_reduced_cols(train_fe)
    print(f"  optimized cols (gru/attn 입력): {len(opt)}")
    print(f"  reduced cols (lgb/cb/xgb 입력): {len(red)}")

    OUT.parent.mkdir(exist_ok=True)
    test_fe.to_parquet(OUT, index=False)
    print(f"saved → {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
