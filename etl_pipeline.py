import pandas as pd
from sqlalchemy import create_engine
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
import time

# 1. EXTRACT stage: players we will use to fetch data
player_names = ["LeBron James", "Russell Westbrook", "Stephen Curry"]

# Empty DataFrame to collect all data
all_stats = pd.DataFrame()

for name in player_names:
    # A) Find the player's ID (the API uses ID, not name)
    nba_players = players.get_active_players()
    matched_players = [player for player in nba_players if player['full_name'] == name]
    if not matched_players:
        raise ValueError(f"Player not found: {name}. Please check the names.")
    player_info = matched_players[0]
    player_id = player_info['id']
    
    # B) Download the player's career statistics from the API
    print(f"Downloading data for {name} (ID: {player_id})...")
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    
    # Convert the returned data to a Pandas DataFrame
    df = career.get_data_frames()[0]
    
    # 2. TRANSFORM stage: add a name column so we can identify the player
    df['PLAYER_NAME'] = name
    
    # Append the data to our main DataFrame
    all_stats = pd.concat([all_stats, df], ignore_index=True)
    
    # IMPORTANT: wait 1.5 seconds to avoid getting blocked by NBA servers
    time.sleep(1.5)

print("\nData download complete!")
print(f"Collected a total of {len(all_stats)} season records.")

# 3. LOAD stage: write the data to an SQL database
print("Saving data to SQLite database...")

# Creates a file named 'nba_database.db' in the project folder
engine = create_engine('sqlite:///nba_database.db')

# Write the DataFrame as a SQL table
all_stats.to_sql('player_careers', con=engine, if_exists='replace', index=False)

print("\nAll data has been successfully saved to 'nba_database.db'.")