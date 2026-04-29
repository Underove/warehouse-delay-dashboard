"""Phase 0 검증용 데이터 로더. 직접 실행하면 데이터 무결성 리포트를 출력한다."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LAYOUT_CSV, SEQ_LEN, SUBMISSION_CSV, TARGET, TEST_CSV, TRAIN_CSV


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def load_layout() -> pd.DataFrame:
    return pd.read_csv(LAYOUT_CSV)


def load_submission() -> pd.DataFrame:
    return pd.read_csv(SUBMISSION_CSV)


def report() -> None:
    train = load_train()
    test = load_test()
    layout = load_layout()
    submission = load_submission()

    print("=" * 60)
    print("FILE SHAPES")
    print("=" * 60)
    print(f"train       {train.shape}")
    print(f"test        {test.shape}")
    print(f"layout_info {layout.shape}")
    print(f"submission  {submission.shape}")

    print("\n" + "=" * 60)
    print("TARGET CHECK")
    print("=" * 60)
    if TARGET in train.columns:
        print(f"target='{TARGET}'  mean={train[TARGET].mean():.4f}  "
              f"min={train[TARGET].min():.4f}  max={train[TARGET].max():.4f}  "
              f"na={train[TARGET].isna().sum()}")
    else:
        print(f"WARNING: '{TARGET}' not in train columns")
        print(f"  candidate cols: {[c for c in train.columns if 'delay' in c.lower()]}")

    print("\n" + "=" * 60)
    print("LAYOUT COVERAGE")
    print("=" * 60)
    train_layouts = set(train["layout_id"].unique()) if "layout_id" in train.columns else set()
    test_layouts = set(test["layout_id"].unique()) if "layout_id" in test.columns else set()
    print(f"train unique layout_id  {len(train_layouts)}")
    print(f"test  unique layout_id  {len(test_layouts)}")
    print(f"test-only layouts       {len(test_layouts - train_layouts)}")
    print(f"layout_info rows        {len(layout)}")

    print("\n" + "=" * 60)
    print("SEQUENCE LENGTH (per scenario_id)")
    print("=" * 60)
    if "scenario_id" in train.columns:
        ts_counts = train.groupby("scenario_id").size()
        print(f"train  min={ts_counts.min()}  max={ts_counts.max()}  median={int(ts_counts.median())}")
        print(f"expected SEQ_LEN={SEQ_LEN}  match={ts_counts.eq(SEQ_LEN).all()}")
    else:
        print("WARNING: 'scenario_id' not in train columns")

    print("\n" + "=" * 60)
    print("MISSING VALUES (top 10)")
    print("=" * 60)
    na = train.isna().sum().sort_values(ascending=False)
    print(na[na > 0].head(10) if (na > 0).any() else "no missing values in train")


if __name__ == "__main__":
    report()
