
from pathlib import Path
import json
import urllib.request
import pandas as pd

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ESPN historical World Cup scoreboard
URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/soccer/fifa.world/scoreboard"
    "?dates=20260611-20260719&limit=200"
)

OUTPUT_CSV = RAW_DIR / "world_cup_2026_matches.csv"
OUTPUT_JSON = RAW_DIR / "espn_2026_response.json"

print("Downloading FIFA World Cup 2026 results...", flush=True)

request = urllib.request.Request(
    URL,
    headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    },
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
except Exception as error:
    print(f"Download failed: {error}")
    raise SystemExit(1)

# Preserve the original response for verification
with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=2, ensure_ascii=False)

events = data.get("events", [])
print(f"Events returned by ESPN: {len(events)}")

records = []

for event in events:
    status = event.get("status", {}).get("type", {})

    # Include only completed matches
    if not status.get("completed", False):
        continue

    competitions = event.get("competitions", [])
    if not competitions:
        continue

    competition = competitions[0]
    competitors = competition.get("competitors", [])

    if len(competitors) != 2:
        continue

    # Preserve ESPN's home/away designation
    home = next(
        (team for team in competitors
         if team.get("homeAway") == "home"),
        None,
    )
    away = next(
        (team for team in competitors
         if team.get("homeAway") == "away"),
        None,
    )

    if home is None or away is None:
        continue

    def team_name(team):
        info = team.get("team", {})
        return (
            info.get("displayName")
            or info.get("name")
            or info.get("shortDisplayName")
        )

    # Stage is retained for later inspection
    stage = (
        competition.get("type", {}).get("abbreviation")
        or event.get("season", {}).get("type", "Unknown")
    )

    records.append({
        "match_id": str(event.get("id")),
        "date": event.get("date", "")[:10],
        "stage": str(stage),
        "team_1": team_name(home),
        "team_2": team_name(away),
        "goals_team_1": home.get("score"),
        "goals_team_2": away.get("score"),
    })

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No completed matches were found.")

df = df.drop_duplicates(subset=["match_id"])
df = df.sort_values(["date", "match_id"])

print(f"Completed unique matches: {len(df)}")

if len(df) != 104:
    print(
        "WARNING: Expected 104 matches. "
        "The CSV will not be overwritten."
    )
    print(df.head(10).to_string(index=False))
    raise SystemExit(1)

# Basic validation before overwriting the template
required = [
    "match_id", "date", "stage",
    "team_1", "team_2",
    "goals_team_1", "goals_team_2",
]

if df[required].isna().any().any():
    raise ValueError("Missing required match information.")

for col in ["goals_team_1", "goals_team_2"]:
    df[col] = pd.to_numeric(df[col], errors="raise")
    if (df[col] < 0).any() or (df[col] % 1 != 0).any():
        raise ValueError(f"Invalid score in {col}")
    df[col] = df[col].astype(int)

df.to_csv(OUTPUT_CSV, index=False)

print("\nDataset downloaded successfully!")
print(f"Saved to: {OUTPUT_CSV}")
print("\nFirst five matches:")
print(df.head().to_string(index=False))
