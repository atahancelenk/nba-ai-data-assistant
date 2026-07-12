import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain.tools import tool

load_dotenv()

@tool
def get_elo_leaderboard() -> str:
    """Returns the current Elo rating leaderboard for all tracked players, ranked highest to lowest."""
    import sqlite3
    conn = sqlite3.connect('nba_database.db')
    df = pd.read_sql("SELECT PLAYER_NAME, ELO_RATING, GAMES_PLAYED FROM player_elo ORDER BY ELO_RATING DESC", conn)
    conn.close()
    lines = [f"{i+1}. {row.PLAYER_NAME}: {row.ELO_RATING:.0f} Elo ({row.GAMES_PLAYED} head-to-head games)"
             for i, row in enumerate(df.itertuples())]
    return "\n".join(lines)

@tool
def predict_player_points(gp: int, mpg: float, fga_pg: float, fta_pg: float) -> str:
    """Predicts the points per game (PPG) a player will score given GP, minutes per game, FGA per game, and FTA per game."""
    try:
        model = joblib.load('nba_points_predictor.joblib')
        input_data = pd.DataFrame([[gp, mpg, fga_pg, fta_pg]], columns=['GP', 'MPG', 'FGA_PG', 'FTA_PG'])
        prediction = model.predict(input_data)[0]
        return f"Predicted Points Per Game: {prediction:.2f}"
    except Exception as e:
        return f"Error making prediction: {e}"

db = SQLDatabase.from_uri("sqlite:///nba_database.db")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling", 
    extra_tools=[predict_player_points, get_elo_leaderboard],
    verbose=False
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        database_metadata_hint = (
        "Database Hint: You have access to a table named 'player_careers'. "
        "This table contains NBA player statistics. The column 'PLAYER_NAME' contains "
        "the exact text names of the players, such as 'LeBron James', 'Russell Westbrook', and 'Stephen Curry'. "
        "Always query 'player_careers' and filter by 'PLAYER_NAME' when asked about a player. "
        "You also have 'player_elo' (current ELO_RATING and GAMES_PLAYED per PLAYER_NAME) "
        "and 'player_elo_history' (one row per head-to-head game showing RATING_BEFORE, "
        "RATING_AFTER, GAME_DATE, OPPONENT_NAME — use this for Elo trend/history questions). "
        )
        
        strict_prompt = (
            database_metadata_hint +
            request.message +
            " (Answer concisely. For predictions, respond in 1-2 sentences mentioning the player name and predicted value. "
            "For career stats, respond with a brief intro sentence followed by a table of per-game averages. "
            "For single season questions, respond with a brief intro sentence and then show ALL the key stats "
            "for that season in a table: SEASON_ID, GP, MIN, PTS, REB, AST, STL, BLK, FG_PCT, FG3_PCT, FT_PCT.)"
        )
        
        response = agent_executor.invoke({"input": strict_prompt})
        return {"reply": response["output"]}
    
    except Exception as e:
        print(f"\n" + "="*40 + f"\n!!! BACKEND ERROR !!!\n{str(e)}\n" + "="*40 + "\n")
        raise HTTPException(status_code=500, detail=str(e))