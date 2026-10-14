
from pathlib import Path
import csv
import re
import time
import urllib.request

import pandas as pd
import pymupdf


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "matches.csv"
OUTPUT_FILE = BASE_DIR / "results" / "match_score_verification.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "match_id", "home_team", "away_team",
    "home_score", "away_score", "source_url"
]

missing = set(required_columns) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")


def normalise(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def extract_score(pdf_text, home_team, away_team):
    # The PDF cover normally starts with:
    # Home Team 2 - 0 Away Team
    lines = [
        line.strip()
        for line in pdf_text.splitlines()
        if line.strip()
    ]

    home = normalise(home_team)
    away = normalise(away_team)

    for line in lines[:15]:
        match = re.match(
            r"^(.+?)\s+(\d{1,2})\s*[-–—]\s*(\d{1,2})\s+(.+)$",
            line
        )

        if not match:
            continue

        pdf_home, home_goals, away_goals, pdf_away = match.groups()

        if (
            normalise(pdf_home) == home
            and normalise(pdf_away) == away
        ):
            return int(home_goals), int(away_goals)

    return None, None


results = []

print(f"Checking {len(df)} FIFA match reports...", flush=True)

for index, row in df.iterrows():
    match_id = str(row["match_id"])
    status = "ERROR"
    pdf_home_score = None
    pdf_away_score = None
    details = ""

    try:
        request = urllib.request.Request(
            str(row["source_url"]),
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            pdf_bytes = response.read()

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        first_page_text = document[0].get_text()
        document.close()

        pdf_home_score, pdf_away_score = extract_score(
            first_page_text,
            row["home_team"],
            row["away_team"]
        )

        if pdf_home_score is None:
            status = "REVIEW"
            details = "Score not found in expected first-page format"
        elif (
            pdf_home_score == int(row["home_score"])
            and pdf_away_score == int(row["away_score"])
        ):
            status = "MATCH"
            details = "PDF score agrees with CSV"
        else:
            status = "MISMATCH"
            details = "PDF score differs from CSV"

    except Exception as error:
        details = str(error)

    results.append({
        "match_id": match_id,
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "csv_home_score": row["home_score"],
        "csv_away_score": row["away_score"],
        "pdf_home_score": pdf_home_score,
        "pdf_away_score": pdf_away_score,
        "status": status,
        "details": details,
        "source_url": row["source_url"]
    })

    print(
        f"{index + 1}/{len(df)} | {match_id} | {status}",
        flush=True
    )

    # Save progress after each match.
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    time.sleep(0.2)


verification = pd.DataFrame(results)

print("\nVerification summary:")
print(verification["status"].value_counts().to_string())
print(f"\nReport saved to: {OUTPUT_FILE}")

if (verification["status"] != "MATCH").any():
    print("\nSome matches require further investigation.")
else:
    print("\nAll match scores agree with their PDF reports.")
