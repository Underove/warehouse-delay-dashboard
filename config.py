from pathlib import Path

TARGET = "avg_delay_minutes_next_30m"
SEQ_LEN = 25
LAYOUT_TYPES = ["narrow", "grid", "hybrid", "hub_spoke"]

RISK_THRESHOLDS = {
    "normal": 15,
    "warning": 22,
    "critical": 30,
}

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent
MODEL_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
LAYOUT_CSV = DATA_DIR / "layout_info.csv"
SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
