import pandas as pd
from sqlalchemy import create_engine
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
from db import engine
import time

<<<<<<< HEAD
=======
engine = create_engine('sqlite:///nba_database.db')

>>>>>>> 5f212fea3cbb6ead9a76e27fc63c99d137366a74
active_players = players.get_active_players()
print(f"Found {len(active_players)} active players.")

# Resume support: skip players already pulled, in case this gets interrupted
try:
    existing_ids = pd.read_sql("SELECT DISTINCT PLAYER_ID FROM player_careers", engine)['PLAYER_ID'].tolist()
except Exception:
    existing_ids = []

for player in active_players:
    player_id = player['id']
    name = player['full_name']

    if player_id in existing_ids:
        continue

    for attempt in range(3):
        try:
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
            df = career.get_data_frames()[0]
            df['PLAYER_NAME'] = name
            df.to_sql('player_careers', con=engine, if_exists='append', index=False)
            print(f"Saved {name}")
            break
        except Exception as e:
            print(f"Retry {attempt+1} for {name}: {e}")
            time.sleep(5 * (attempt + 1))

    time.sleep(1.5)

print("Done.")