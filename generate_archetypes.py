"""
generate_archetypes.py

Assigns a "2K-style" build archetype to every player in the DB based on their
most recent season's box-score rate stats, using percentile thresholds
computed across that same season's roster (not fixed magic numbers).

This mirrors the Elo system's design philosophy: real data, transparent
thresholds, documented limitations -- no hand-picked labels.

Run this AFTER etl_pipeline.py (and ideally after the active-roster expansion,
since percentiles are only meaningful across a real population).

Usage:
    python generate_archetypes.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

DB_PATH = "sqlite:///nba_database.db"

# ─────────────────────────────────────────────────────────────
# Archetype taxonomy
# ─────────────────────────────────────────────────────────────
ARCHETYPES = [
    "Sharpshooter",
    "3-and-D Wing",
    "Two-Way Slasher",
    "Slasher",
    "Playmaker",
    "Rim Protector",
    "Stretch Big",
    "Glass Cleaner",
    "Lockdown Defender",
    "Combo Scorer",  # fallback
]

# Percentile cut lines. Tune these once you see real distributions.
P_HIGH = 0.75
P_MID = 0.60


def load_latest_season_per_player(engine):
    df = pd.read_sql("SELECT * FROM player_careers", con=engine)

    # Keep only real season rows (drop career-total rows nba_api sometimes includes)
    if "SEASON_ID" in df.columns:
        df = df[df["SEASON_ID"].notna()]

    # For each player, find their most recent season
    latest_season = df.groupby("PLAYER_NAME")["SEASON_ID"].transform("max")
    df = df[df["SEASON_ID"] == latest_season].copy()

    # If a player has multiple rows for that season (mid-season trade),
    # prefer the 'TOT' row if present, otherwise sum the partial-team rows.
    def resolve_group(g):
        if "TEAM_ABBREVIATION" in g.columns and (g["TEAM_ABBREVIATION"] == "TOT").any():
            return g[g["TEAM_ABBREVIATION"] == "TOT"].iloc[0]
        numeric_cols = g.select_dtypes(include="number").columns
        summed = g[numeric_cols].sum()
        result = g.iloc[0].copy()
        result[numeric_cols] = summed
        return result

    resolved = df.groupby("PLAYER_NAME", group_keys=False).apply(resolve_group)
    return resolved.reset_index(drop=True)


def compute_features(df):
    df = df.copy()

    # Guard against divide-by-zero for low-usage players
    safe_fga = df["FGA"].replace(0, np.nan)
    safe_min = df["MIN"].replace(0, np.nan)
    safe_fg3a = df["FG3A"].replace(0, np.nan)

    df["FG3A_RATE"] = (df["FG3A"] / safe_fga).fillna(0)
    df["FG3_PCT"] = (df["FG3M"] / safe_fg3a).fillna(0)
    df["FT_RATE"] = (df["FTA"] / safe_fga).fillna(0)
    df["AST_PER36"] = (df["AST"] / safe_min * 36).fillna(0)
    df["REB_PER36"] = (df["REB"] / safe_min * 36).fillna(0)
    df["STOCKS_PER36"] = ((df["STL"] + df["BLK"]) / safe_min * 36).fillna(0)
    df["BLK_PER36"] = (df["BLK"] / safe_min * 36).fillna(0)
    df["USG_PROXY"] = (
        (df["FGA"] + 0.44 * df["FTA"] + df.get("TOV", 0)) / safe_min
    ).fillna(0)

    return df


def add_percentiles(df):
    feature_cols = [
        "FG3A_RATE", "FG3_PCT", "FT_RATE", "AST_PER36",
        "REB_PER36", "STOCKS_PER36", "BLK_PER36", "USG_PROXY",
    ]
    for col in feature_cols:
        df[f"{col}_PCT"] = df[col].rank(pct=True)
    return df


def classify(row):
    """Top-down gated decision tree. Defensive fits checked first since
    2-way builds are rarer and more specific; offensive skill next;
    fallback ensures nobody is left unclassified."""

    stocks = row["STOCKS_PER36_PCT"]
    fg3a = row["FG3A_RATE_PCT"]
    fg3pct = row["FG3_PCT_PCT"]
    ft_rate = row["FT_RATE_PCT"]
    ast = row["AST_PER36_PCT"]
    reb = row["REB_PER36_PCT"]
    blk = row["BLK_PER36_PCT"]
    usg = row["USG_PROXY_PCT"]

    # Defense-forward builds
    if stocks >= P_HIGH and fg3a >= P_MID:
        return "3-and-D Wing"
    if stocks >= P_HIGH and ft_rate >= P_MID and usg >= P_MID:
        return "Two-Way Slasher"
    if stocks >= P_HIGH:
        return "Lockdown Defender"

    # Big-man builds
    if reb >= P_HIGH and blk >= P_MID and fg3a >= P_MID:
        return "Stretch Big"
    if reb >= P_HIGH and blk >= P_MID:
        return "Rim Protector"
    if reb >= P_HIGH:
        return "Glass Cleaner"

    # Offensive skill builds
    if fg3a >= P_HIGH and fg3pct >= P_MID:
        return "Sharpshooter"
    if ast >= P_HIGH:
        return "Playmaker"
    if ft_rate >= P_HIGH and usg >= P_MID:
        return "Slasher"

    return "Combo Scorer"


def main():
    engine = create_engine(DB_PATH)

    print("Loading most recent season per player...")
    df = load_latest_season_per_player(engine)
    print(f"Resolved {len(df)} players.")

    print("Computing rate stats...")
    df = compute_features(df)

    print("Computing roster percentiles...")
    df = add_percentiles(df)

    print("Classifying archetypes...")
    df["ARCHETYPE"] = df.apply(classify, axis=1)

    output_cols = [
        "PLAYER_NAME", "SEASON_ID", "ARCHETYPE",
        "FG3A_RATE", "FG3_PCT", "FT_RATE",
        "AST_PER36", "REB_PER36", "STOCKS_PER36", "USG_PROXY",
    ]
    result = df[output_cols].rename(columns={"SEASON_ID": "SEASON"})

    print("Saving to 'player_archetypes' table...")
    result.to_sql("player_archetypes", con=engine, if_exists="replace", index=False)

    print("\nDistribution:")
    print(result["ARCHETYPE"].value_counts())
    print("\nDone. 'player_archetypes' table written to nba_database.db")


if __name__ == "__main__":
    main()