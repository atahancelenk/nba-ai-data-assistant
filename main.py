import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from db import engine, DATABASE_URL

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain.tools import tool

load_dotenv()

def _predict_from_model(filename: str, gp: int, mpg: float, fga_pg: float, fta_pg: float) -> float:
    """
    Loads a joblib file saved by train_model.py — a dict of
    {'model', 'model_name', 'features', 'target'} rather than a bare
    sklearn model — and runs a prediction with the given inputs.

    Building the input row from `features` (instead of assuming column
    order) keeps this working even if train_model.py's FEATURE_COLUMNS
    order ever changes.
    """
    saved = joblib.load(filename)
    model = saved['model']
    features = saved['features']

    available_inputs = {'GP': gp, 'MPG': mpg, 'FGA_PG': fga_pg, 'FTA_PG': fta_pg}
    input_data = pd.DataFrame([[available_inputs[f] for f in features]], columns=features)

    return model.predict(input_data)[0]

@tool
def get_elo_leaderboard() -> str:
    """Returns the current Elo rating leaderboard for all tracked players, ranked highest to lowest."""
    df = pd.read_sql('SELECT "PLAYER_NAME", "ELO_RATING", "GAMES_PLAYED" FROM player_elo ORDER BY "ELO_RATING" DESC', engine)
    lines = [f"{i+1}. {row.PLAYER_NAME}: {row.ELO_RATING:.0f} Elo ({row.GAMES_PLAYED} head-to-head games)"
             for i, row in enumerate(df.itertuples())]
    return "\n".join(lines)

@tool
def predict_player_points(gp: int, mpg: float, fga_pg: float, fta_pg: float) -> str:
    """Predicts points per game (PPG) for a hypothetical player given GP (games played,
    e.g. 70), MPG (minutes per game, e.g. 34.5), FGA_PG (field goal attempts per game,
    e.g. 16.2), and FTA_PG (free throw attempts per game, e.g. 5.1)."""
    try:
        prediction = _predict_from_model('nba_points_predictor.joblib', gp, mpg, fga_pg, fta_pg)
        return f"Predicted Points Per Game: {prediction:.2f}"
    except Exception as e:
        return f"Error making prediction: {e}"

@tool
def predict_player_rebounds(gp: int, mpg: float, fga_pg: float, fta_pg: float) -> str:
    """Predicts rebounds per game (RPG) for a hypothetical player given GP (games played,
    e.g. 70), MPG (minutes per game, e.g. 34.5), FGA_PG (field goal attempts per game,
    e.g. 16.2), and FTA_PG (free throw attempts per game, e.g. 5.1)."""
    try:
        prediction = _predict_from_model('nba_rebounds_predictor.joblib', gp, mpg, fga_pg, fta_pg)
        return f"Predicted Rebounds Per Game: {prediction:.2f}"
    except Exception as e:
        return f"Error making prediction: {e}"

@tool
def predict_player_assists(gp: int, mpg: float, fga_pg: float, fta_pg: float) -> str:
    """Predicts assists per game (APG) for a hypothetical player given GP (games played,
    e.g. 70), MPG (minutes per game, e.g. 34.5), FGA_PG (field goal attempts per game,
    e.g. 16.2), and FTA_PG (free throw attempts per game, e.g. 5.1)."""
    try:
        prediction = _predict_from_model('nba_assists_predictor.joblib', gp, mpg, fga_pg, fta_pg)
        return f"Predicted Assists Per Game: {prediction:.2f}"
    except Exception as e:
        return f"Error making prediction: {e}"

db = SQLDatabase.from_uri(DATABASE_URL)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling",
    extra_tools=[
        predict_player_points,
        predict_player_rebounds,
        predict_player_assists,
        get_elo_leaderboard,
    ],
    verbose=False
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

@app.get("/players")
async def list_players():
    df = pd.read_sql('SELECT DISTINCT "PLAYER_NAME" FROM player_careers ORDER BY "PLAYER_NAME"', con=engine)
    return {"players": df['PLAYER_NAME'].tolist()}

@app.get("/elo/leaderboard-chart")
async def elo_leaderboard_chart(limit: int = 15):
    """Top-N Elo leaderboard as chart-ready JSON (labels + data), highest first."""
    query = text(
        'SELECT "PLAYER_NAME", "ELO_RATING" FROM player_elo '
        'ORDER BY "ELO_RATING" DESC LIMIT :limit'
    )
    df = pd.read_sql(query, con=engine, params={"limit": limit})
    if df.empty:
        raise HTTPException(status_code=404, detail="No Elo ratings found. Has build_elo_ratings.py been run?")
    return {
        "labels": df['PLAYER_NAME'].tolist(),
        "data": [round(v) for v in df['ELO_RATING'].tolist()],
    }

@app.get("/elo/history-chart/{player_name}")
async def elo_history_chart(player_name: str):
    """Chronological Elo rating history for one player, as chart-ready JSON."""
    query = text(
        'SELECT "GAME_DATE", "RATING_AFTER" FROM player_elo_history '
        'WHERE "PLAYER_NAME" = :player_name ORDER BY "GAME_DATE" ASC'
    )
    df = pd.read_sql(query, con=engine, params={"player_name": player_name})
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No Elo history found for '{player_name}'. Check the exact spelling of the player's name.",
        )
    return {
        "labels": df['GAME_DATE'].astype(str).tolist(),
        "data": [round(v) for v in df['RATING_AFTER'].tolist()],
    }

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        database_metadata_hint = (
            "IMPORTANT: All column names in this database are UPPERCASE (e.g. PLAYER_NAME, GP, PTS, "
            "ELO_RATING). This is a Postgres database, and Postgres is case-sensitive for identifiers "
            "that aren't all-lowercase. You MUST wrap every column name in double quotes exactly as "
            'shown, e.g. SELECT "PLAYER_NAME", "PTS" FROM player_careers -- an unquoted or '
            "lowercased column name will fail with a 'column does not exist' error. Table names "
            "(player_careers, player_elo, player_elo_history, player_archetypes) are lowercase and "
            "do not need quoting. "
            "You have access to a table named 'player_careers' containing NBA player "
            "season statistics. The 'PLAYER_NAME' column contains the player's full name "
            "as text, e.g. 'LeBron James'. Match names as closely as possible. "
            "Always query 'player_careers' and filter by 'PLAYER_NAME' when asked about career stats. "
            "You also have 'player_elo' (current ELO_RATING and GAMES_PLAYED per PLAYER_NAME), "
            "'player_elo_history' (per-game Elo history), and 'player_archetypes' "
            "(PLAYER_NAME, SEASON, ARCHETYPE, and underlying rate stats like FG3A_RATE, AST_PER36, "
            "REB_PER36, STOCKS_PER36 — use this when asked about a player's build, playstyle, or archetype). "
            "For predictions, you have three separate tools: predict_player_points (PPG), "
            "predict_player_rebounds (RPG), and predict_player_assists (APG) — each takes the same "
            "GP, MPG, FGA_PG, FTA_PG inputs. If the user asks for a full predicted stat line, call "
            "all three tools and combine the results. "
        )

        strict_prompt = (
            database_metadata_hint +
            request.message +
            " (Answer concisely. For predictions, respond in 1-2 sentences mentioning the player name and predicted value(s). "
            "For career stats, respond with a brief intro sentence followed by a table of per-game averages. "
            "For single season questions, respond with a brief intro sentence and then show ALL the key stats "
            "for that season in a table: SEASON_ID, GP, MIN, PTS, REB, AST, STL, BLK, FG_PCT, FG3_PCT, FT_PCT. "
            "Round all per-game and percentage stats (PPG, APG, RPG, FG_PCT, FG3_PCT, FT_PCT, etc.) to 1 decimal place. "
            "Elo ratings should always be shown as whole numbers with no decimals.)"
        )

        response = agent_executor.invoke({"input": strict_prompt})
        return {"reply": response["output"]}

    except Exception as e:
        print(f"\n" + "="*40 + f"\n!!! BACKEND ERROR !!!\n{str(e)}\n" + "="*40 + "\n")
        raise HTTPException(status_code=500, detail=str(e))