"""
build_elo_ratings.py

Pulls per-game logs for the three tracked players, finds actual head-to-head
games (same GAME_ID, different players), scores each game with Hollinger's
Game Score, and runs a chronological Elo update across the shared rating pool.

Run this AFTER etl_pipeline.py, since it reuses the SEASON_ID list already
stored in player_careers to know which seasons to pull game logs for.
"""

import time
import pandas as pd
from sqlalchemy import create_engine
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

PLAYER_NAMES = ["LeBron James", "Russell Westbrook", "Stephen Curry"]
K_FACTOR = 20
STARTING_ELO = 1500

engine = create_engine('sqlite:///nba_database.db')


def get_player_id(name: str) -> int:
    matches = [p for p in players.get_active_players() + players.get_inactive_players()
               if p['full_name'] == name]
    if not matches:
        raise ValueError(f"Player not found: {name}")
    return matches[0]['id']


def fetch_game_logs() -> pd.DataFrame:
    """Pull every regular-season game log row for each tracked player."""
    careers = pd.read_sql("SELECT DISTINCT PLAYER_NAME, SEASON_ID FROM player_careers", engine)

    all_logs = pd.DataFrame()
    for name in PLAYER_NAMES:
        player_id = get_player_id(name)
        seasons = careers[careers['PLAYER_NAME'] == name]['SEASON_ID'].dropna().unique()

        for season in seasons:
            print(f"Fetching game log: {name} / {season}")
            try:
                log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
                df = log.get_data_frames()[0]
                df['PLAYER_NAME'] = name
                all_logs = pd.concat([all_logs, df], ignore_index=True)
            except Exception as e:
                print(f"  Skipped {name}/{season}: {e}")
            time.sleep(1.5)

    all_logs.to_sql('player_game_logs', con=engine, if_exists='replace', index=False)
    print(f"\nSaved {len(all_logs)} total game log rows to player_game_logs.")
    return all_logs


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


def find_head_to_head(logs: pd.DataFrame) -> pd.DataFrame:
    """Self-join on GAME_ID to find games where two tracked players both played."""
    logs = logs.copy()
    logs['GAME_SCORE'] = logs.apply(compute_game_score, axis=1)

    merged = logs.merge(logs, on='Game_ID', suffixes=('_A', '_B'))
    merged = merged[merged['PLAYER_NAME_A'] < merged['PLAYER_NAME_B']]  # dedupe A-B / B-A, drop self-pairs

    h2h = merged[[
        'Game_ID', 'GAME_DATE_A', 'PLAYER_NAME_A', 'PLAYER_NAME_B',
        'GAME_SCORE_A', 'GAME_SCORE_B'
    ]].rename(columns={'GAME_DATE_A': 'GAME_DATE'})

    h2h = h2h.sort_values('GAME_DATE').reset_index(drop=True)
    print(f"\nFound {len(h2h)} head-to-head games among the tracked players.")
    return h2h


def run_elo(h2h: pd.DataFrame) -> tuple[dict, list]:
    ratings = {name: STARTING_ELO for name in PLAYER_NAMES}
    games_played = {name: 0 for name in PLAYER_NAMES}
    history = []

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
        ratings[a] = ratings[a] + K_FACTOR * (actual_a - expected_a)
        ratings[b] = ratings[b] + K_FACTOR * (actual_b - expected_b)
        games_played[a] += 1
        games_played[b] += 1

        history.append({
            'GAME_ID': game['Game_ID'], 'GAME_DATE': game['GAME_DATE'],
            'PLAYER_NAME': a, 'OPPONENT_NAME': b,
            'GAME_SCORE': gs_a, 'OPPONENT_GAME_SCORE': gs_b,
            'RATING_BEFORE': rating_a_before, 'RATING_AFTER': ratings[a],
        })
        history.append({
            'GAME_ID': game['Game_ID'], 'GAME_DATE': game['GAME_DATE'],
            'PLAYER_NAME': b, 'OPPONENT_NAME': a,
            'GAME_SCORE': gs_b, 'OPPONENT_GAME_SCORE': gs_a,
            'RATING_BEFORE': rating_b_before, 'RATING_AFTER': ratings[b],
        })

    return {name: {'ELO_RATING': ratings[name], 'GAMES_PLAYED': games_played[name]}
            for name in PLAYER_NAMES}, history


def main():
    logs = fetch_game_logs()
    h2h = find_head_to_head(logs)

    if h2h.empty:
        print("No head-to-head games found — check GAME_ID matching / season coverage.")
        return

    final_ratings, history = run_elo(h2h)

    elo_df = pd.DataFrame([
        {'PLAYER_NAME': name, **stats} for name, stats in final_ratings.items()
    ])
    elo_df.to_sql('player_elo', con=engine, if_exists='replace', index=False)

    history_df = pd.DataFrame(history)
    history_df.to_sql('player_elo_history', con=engine, if_exists='replace', index=False)

    print("\n--- Final Elo Ratings ---")
    print(elo_df.to_string(index=False))


if __name__ == "__main__":
    main()