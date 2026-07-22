"""
build_elo_ratings.py

Computes Elo ratings for all active NBA players based on REAL head-to-head
games (same GAME_ID, opposing teams). Scales to ~500 players by pulling
game logs league-wide, one call per season, instead of one call per player
per season.

This is a one-time (or occasional-refresh) batch job. It writes final
results to 'player_elo' and 'player_elo_history' tables in nba_database.db.
Your chat endpoint should only ever READ those tables, never recompute
Elo live.
"""

import os
import time
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamelog
from db import engine

K_FACTOR = 20
STARTING_ELO = 1500

# Adjust the end year as seasons roll over. Format: 'YYYY-YY'.
START_YEAR = 2003   # earliest season worth checking (covers longest-tenured active players)
END_YEAR = 2025      # season START year of the most recent completed/current season

CACHE_DIR = "elo_cache"

def season_strings(start_year: int, end_year: int) -> list:
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, end_year + 1)]

def get_active_player_ids() -> dict:
    """Returns {PLAYER_ID: PLAYER_NAME} for the current active roster (~500 players)."""
    active = players.get_active_players()
    return {p['id']: p['full_name'] for p in active}

def fetch_season_log(season: str) -> pd.DataFrame:
    """
    Pulls the ENTIRE league's player game logs for one season in a single API call.
    Cached locally so reruns (e.g. after fixing Elo logic) don't hit the API again.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"league_log_{season}.csv")

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    print(f"Fetching league-wide game log for {season}...")
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star='Regular Season',
        player_or_team_abbreviation='P'
    )
    df = log.get_data_frames()[0]
    df.to_csv(cache_path, index=False)
    time.sleep(1.5)
    return df

def compute_game_score(row: pd.Series) -> float:
    return (
        row['PTS']
        + 0.4 * row['FGM']
        - 0.7 * row['FGA']
        - 0.4 * (row['FTA'] - row['FTM'])
        + 0.7 * row['OREB']
        + 0.3 * row['DREB']
        + row['STL']
        + 0.7 * row['AST']
        + 0.7 * row['BLK']
        - 0.4 * row['PF']
        - row['TOV']
    )

def head_to_head_for_season(season_df: pd.DataFrame, active_ids: set, active_map: dict) -> pd.DataFrame:
    """
    Filters to tracked active players, then self-joins on GAME_ID to find
    real head-to-head games, excluding teammate pairs (same TEAM_ID).
    """
    df = season_df[season_df['PLAYER_ID'].isin(active_ids)].copy()
    if df.empty:
        return pd.DataFrame()

    # LeagueGameLog's PLAYER_NAME column sometimes drops diacritics (e.g.
    # "Jonas Valanciunas" instead of "Jonas Valančiūnas"), which would silently
    # create a second, unmatched key in the ratings dict. Always resolve the
    # canonical name by PLAYER_ID instead of trusting the log's name string.
    df['PLAYER_NAME'] = df['PLAYER_ID'].map(active_map)

    df['GAME_SCORE'] = df.apply(compute_game_score, axis=1)

    keep_cols = ['GAME_ID', 'GAME_DATE', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'GAME_SCORE']
    df = df[keep_cols]

    merged = df.merge(df, on='GAME_ID', suffixes=('_A', '_B'))
    merged = merged[
        (merged['TEAM_ID_A'] != merged['TEAM_ID_B']) &      # exclude teammates
        (merged['PLAYER_ID_A'] < merged['PLAYER_ID_B'])     # exclude self + dedupe A-B/B-A
    ]

    h2h = merged[[
        'GAME_ID', 'GAME_DATE_A', 'PLAYER_NAME_A', 'PLAYER_NAME_B',
        'GAME_SCORE_A', 'GAME_SCORE_B'
    ]].rename(columns={'GAME_DATE_A': 'GAME_DATE'})

    return h2h.sort_values('GAME_DATE').reset_index(drop=True)

def apply_elo_updates(h2h: pd.DataFrame, ratings: dict, games_played: dict, history: list):
    """Mutates ratings/games_played/history in place, processing games in order."""
    for _, game in h2h.iterrows():
        a, b = game['PLAYER_NAME_A'], game['PLAYER_NAME_B']
        gs_a, gs_b = game['GAME_SCORE_A'], game['GAME_SCORE_B']

        if gs_a > gs_b:
            actual_a = 1.0
        elif gs_a < gs_b:
            actual_a = 0.0
        else:
            actual_a = 0.5
        actual_b = 1.0 - actual_a

        expected_a = 1 / (1 + 10 ** ((ratings[b] - ratings[a]) / 400))
        expected_b = 1 - expected_a

        rating_a_before, rating_b_before = ratings[a], ratings[b]
        ratings[a] += K_FACTOR * (actual_a - expected_a)
        ratings[b] += K_FACTOR * (actual_b - expected_b)
        games_played[a] += 1
        games_played[b] += 1

        history.append({
            'GAME_ID': game['GAME_ID'], 'GAME_DATE': game['GAME_DATE'],
            'PLAYER_NAME': a, 'OPPONENT_NAME': b,
            'GAME_SCORE': gs_a, 'OPPONENT_GAME_SCORE': gs_b,
            'RATING_BEFORE': rating_a_before, 'RATING_AFTER': ratings[a],
        })
        history.append({
            'GAME_ID': game['GAME_ID'], 'GAME_DATE': game['GAME_DATE'],
            'PLAYER_NAME': b, 'OPPONENT_NAME': a,
            'GAME_SCORE': gs_b, 'OPPONENT_GAME_SCORE': gs_a,
            'RATING_BEFORE': rating_b_before, 'RATING_AFTER': ratings[b],
        })

def main():
    active_map = get_active_player_ids()
    active_ids = set(active_map.keys())
    active_names = set(active_map.values())
    print(f"Tracking {len(active_names)} active players.")

    ratings = {name: STARTING_ELO for name in active_names}
    games_played = {name: 0 for name in active_names}
    history = []

    seasons = season_strings(START_YEAR, END_YEAR)
    for season in seasons:
        season_df = fetch_season_log(season)
        h2h = head_to_head_for_season(season_df, active_ids, active_map)

        if h2h.empty:
            print(f"  {season}: no head-to-head games among tracked players.")
            continue

        print(f"  {season}: {len(h2h)} head-to-head games.")
        apply_elo_updates(h2h, ratings, games_played, history)

    elo_df = pd.DataFrame([
        {'PLAYER_NAME': name, 'ELO_RATING': ratings[name], 'GAMES_PLAYED': games_played[name]}
        for name in active_names
    ]).sort_values('ELO_RATING', ascending=False)

    elo_df.to_sql('player_elo', con=engine, if_exists='replace', index=False)

    history_df = pd.DataFrame(history)
    history_df.to_sql('player_elo_history', con=engine, if_exists='replace', index=False)

    print("\n--- Top 10 Elo Ratings ---")
    print(elo_df.head(10).to_string(index=False))
    print(f"\nSaved {len(elo_df)} player ratings and {len(history_df)} history rows.")

if __name__ == "__main__":
    main()