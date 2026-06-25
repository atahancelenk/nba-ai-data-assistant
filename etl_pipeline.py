import pandas as pd
from sqlalchemy import create_engine
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
import time

player_names = ["LeBron James", "Russell Westbrook", "Stephen Curry"]

all_stats = pd.DataFrame()

for name in player_names:
    nba_players = players.get_active_players()
    matched_players = [player for player in nba_players if player['full_name'] == name]
    if not matched_players:
        raise ValueError(f"Player not found: {name}. Please check the names.")
    player_info = matched_players[0]
    player_id = player_info['id']
    
    print(f"Downloading data for {name} (ID: {player_id})...")
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    
    df = career.get_data_frames()[0]
    
    df['PLAYER_NAME'] = name
    
    all_stats = pd.concat([all_stats, df], ignore_index=True)
    
    time.sleep(1.5)

print("\nData download complete!")
print(f"Collected a total of {len(all_stats)} season records.")

print("Saving data to SQLite database...")

engine = create_engine('sqlite:///nba_database.db')

all_stats.to_sql('player_careers', con=engine, if_exists='replace', index=False)

print("\nAll data has been successfully saved to 'nba_database.db'.")