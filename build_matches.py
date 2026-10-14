
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "world_cup_2026_matches.csv"
OUTPUT_FILE = PROCESSED_DIR / "matches_validated.csv"

REQUIRED_COLUMNS = [
    "match_id",
    "date",
    "stage",
    "team_1",
    "team_2",
    "goals_team_1",
    "goals_team_2",
]

if not INPUT_FILE.exists():
    template = pd.DataFrame(columns=REQUIRED_COLUMNS)
    template.to_csv(INPUT_FILE, index=False)

    print("Created an empty match-data template:")
    print(INPUT_FILE)
    print("Next, populate it with verified match results.")
    raise SystemExit(0)

df = pd.read_csv(INPUT_FILE)

missing = set(REQUIRED_COLUMNS) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if len(df) != 104:
    raise ValueError(
        f"Expected exactly 104 matches, found {len(df)}."
    )

if df[REQUIRED_COLUMNS].isna().any().any():
    raise ValueError("Missing values found in required columns.")

if df["match_id"].duplicated().any():
    raise ValueError("Duplicate match IDs detected.")

if (df["team_1"] == df["team_2"]).any():
    raise ValueError("A match cannot contain the same team twice.")

df["date"] = pd.to_datetime(df["date"], errors="raise")

for column in ["goals_team_1", "goals_team_2"]:
    df[column] = pd.to_numeric(df[column], errors="raise")

    if (df[column] < 0).any():
        raise ValueError(f"Negative goals found in {column}.")

    if (df[column] % 1 != 0).any():
        raise ValueError(f"Non-integer goals found in {column}.")

    df[column] = df[column].astype(int)

df["goal_difference"] = (
    df["goals_team_1"] - df["goals_team_2"]
)

df = df.sort_values(["date", "match_id"])
df.to_csv(OUTPUT_FILE, index=False)

print("Validation completed successfully.")
print(f"Total matches: {len(df)}")
print(f"Saved: {OUTPUT_FILE}")
print(df.head())
