
from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

INPUT_FILE = RAW_DIR / "matches.csv"
OUTPUT_FILE = RAW_DIR / "world_cup_2026_matches.csv"

# Read the original dataset
df = pd.read_csv(INPUT_FILE)

print("Original dataset loaded successfully.")
print(f"Total matches: {len(df)}")

# Check required source columns
column_mapping = {
    "match_id": "match_id",
    "date": "date",
    "group": "stage",
    "home_team": "team_1",
    "away_team": "team_2",
    "home_score": "goals_team_1",
    "away_score": "goals_team_2",
}

missing = set(column_mapping) - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

# Select and rename the required columns
matches = df[list(column_mapping)].rename(
    columns=column_mapping
)

# Convert dates to standard YYYY-MM-DD format
matches["date"] = pd.to_datetime(
    matches["date"],
    format="%d %B %Y",
    errors="raise"
).dt.strftime("%Y-%m-%d")

# Validate before saving
if len(matches) != 104:
    raise ValueError(
        f"Expected 104 matches, found {len(matches)}."
    )

if matches.isna().any().any():
    raise ValueError("Missing values detected.")

if matches["match_id"].duplicated().any():
    raise ValueError("Duplicate match IDs detected.")

for column in ["goals_team_1", "goals_team_2"]:
    matches[column] = pd.to_numeric(
        matches[column],
        errors="raise"
    )

    if (
        (matches[column] < 0).any()
        or (matches[column] % 1 != 0).any()
    ):
        raise ValueError(f"Invalid scores in {column}.")

    matches[column] = matches[column].astype(int)

# Sort matches chronologically
matches = matches.sort_values(
    ["date", "match_id"]
).reset_index(drop=True)

# Save the converted dataset
matches.to_csv(OUTPUT_FILE, index=False)

print("\nConversion completed successfully!")
print(f"Total matches: {len(matches)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nFirst five matches:")
print(matches.head().to_string(index=False))

print("\nLast five matches:")
print(matches.tail().to_string(index=False))
