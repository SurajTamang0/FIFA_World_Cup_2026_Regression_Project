
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "data/processed/matches_validated.csv"
OUTPUT = BASE / "data/processed"

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

# Only previous matches may contribute to a team's statistics.
history = defaultdict(list)

def get_stats(team):
    games = history[team]
    n = len(games)

    if n == 0:
        return {
            "played": 0,
            "win_rate": 0.0,
            "avg_scored": 0.0,
            "avg_conceded": 0.0,
            "avg_gd": 0.0,
            "scoring_std": 0.0,
            "clean_sheet_rate": 0.0,
        }

    scored = np.array([g[0] for g in games], dtype=float)
    conceded = np.array([g[1] for g in games], dtype=float)

    return {
        "played": n,
        "win_rate": float(np.mean(scored > conceded)),
        "avg_scored": float(scored.mean()),
        "avg_conceded": float(conceded.mean()),
        "avg_gd": float((scored - conceded).mean()),
        "scoring_std": float(scored.std(ddof=0)),
        "clean_sheet_rate": float(np.mean(conceded == 0)),
    }

match_rows = []
team_rows = []

# Process one date at a time so same-day results
# cannot be used to predict another match that day.
for date, daily_matches in df.groupby("date", sort=True):

    updates = []

    for _, match in daily_matches.iterrows():
        t1 = match["team_1"]
        t2 = match["team_2"]

        s1 = get_stats(t1)
        s2 = get_stats(t2)

        knockout = int(
            not str(match["stage"]).startswith("Group")
        )

        # Regression 2.1: exactly 8 predictors
        match_rows.append({
            "match_id": match["match_id"],
            "date": match["date"].strftime("%Y-%m-%d"),
            "team_1": t1,
            "team_2": t2,

            "team1_win_rate": s1["win_rate"],
            "team2_win_rate": s2["win_rate"],
            "team1_avg_gd": s1["avg_gd"],
            "team2_avg_gd": s2["avg_gd"],
            "win_rate_difference":
                s1["win_rate"] - s2["win_rate"],
            "avg_goals_scored_difference":
                s1["avg_scored"] - s2["avg_scored"],
            "team1_previous_matches": s1["played"],
            "knockout_stage": knockout,

            "goal_difference": match["goal_difference"],
        })

        # Regression 2.2: one row for each team
        for team, own, opponent, opp in [
            (t1, s1, t2, s2),
            (t2, s2, t1, s1),
        ]:
            team_rows.append({
                "match_id": match["match_id"],
                "date": match["date"].strftime("%Y-%m-%d"),
                "team": team,
                "opponent": opponent,

                "team_avg_goals_scored":
                    own["avg_scored"],
                "team_avg_goals_conceded":
                    own["avg_conceded"],
                "team_scoring_std":
                    own["scoring_std"],
                "team_clean_sheet_rate":
                    own["clean_sheet_rate"],
                "opponent_avg_goals_conceded":
                    opp["avg_conceded"],
                "opponent_clean_sheet_rate":
                    opp["clean_sheet_rate"],
                "team_previous_matches":
                    own["played"],
                "knockout_stage": knockout,

                "goals_scored": (
                    match["goals_team_1"]
                    if team == t1
                    else match["goals_team_2"]
                ),
            })

        # Save results for updating history after
        # all fixtures on this date are processed.
        updates.append((
            t1, t2,
            int(match["goals_team_1"]),
            int(match["goals_team_2"])
        ))

    for t1, t2, g1, g2 in updates:
        history[t1].append((g1, g2))
        history[t2].append((g2, g1))

match_df = pd.DataFrame(match_rows)
team_df = pd.DataFrame(team_rows)

# Check required observation counts
assert len(match_df) == 104, "Expected 104 match rows"
assert len(team_df) == 208, "Expected 208 team rows"

# Identify exactly eight predictors in each dataset
metadata_21 = {
    "match_id", "date", "team_1", "team_2",
    "goal_difference",
}
metadata_22 = {
    "match_id", "date", "team", "opponent",
    "goals_scored",
}

features_21 = set(match_df.columns) - metadata_21
features_22 = set(team_df.columns) - metadata_22

assert len(features_21) == 8
assert len(features_22) == 8

shared = features_21 & features_22
assert len(shared) <= 4

# Check for missing predictor values
assert not match_df[list(features_21)].isna().any().any()
assert not team_df[list(features_22)].isna().any().any()

file_21 = OUTPUT / "regression_2_1.csv"
file_22 = OUTPUT / "regression_2_2.csv"

match_df.to_csv(file_21, index=False)
team_df.to_csv(file_22, index=False)

print("Feature engineering completed successfully!")
print(f"Regression 2.1 observations: {len(match_df)}")
print(f"Regression 2.1 predictors: {len(features_21)}")
print(f"Regression 2.2 observations: {len(team_df)}")
print(f"Regression 2.2 predictors: {len(features_22)}")
print(f"Shared predictors: {sorted(shared)}")
print(f"Saved: {file_21}")
print(f"Saved: {file_22}")
