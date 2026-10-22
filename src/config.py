from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv"
DATA_PATH = ROOT / "data" / "CollegeDistance.csv"

REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

TARGET = "score"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

NUMERIC_COLS = ["unemp", "wage", "distance", "tuition", "education"]
BINARY_COLS = ["gender", "fcollege", "mcollege", "home", "urban", "income", "region"]
NOMINAL_COLS = ["ethnicity"]

BINARY_MAP = {
    "gender": {"male": 1, "female": 0},
    "fcollege": {"yes": 1, "no": 0},
    "mcollege": {"yes": 1, "no": 0},
    "home": {"yes": 1, "no": 0},
    "urban": {"yes": 1, "no": 0},
    "income": {"high": 1, "low": 0},
    "region": {"west": 1, "other": 0},
}

for d in (REPORTS_DIR, FIG_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
