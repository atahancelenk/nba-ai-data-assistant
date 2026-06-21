import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# LangChain Modules
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

# 1. Kendi ML Aracımız (Öncekiyle aynı)
@tool
def predict_player_points(gp: int, min_played: float, fga: float, fta: float) -> str:
    """Predicts the total points (PTS) a player will score."""
    try:
        model = joblib.load('nba_points_predictor.joblib')
        input_data = pd.DataFrame([[gp, min_played, fga, fta]], columns=['GP', 'MIN', 'FGA', 'FTA'])
        prediction = model.predict(input_data)[0]
        return f"Predicted Total Points: {prediction:.2f}"
    except Exception as e:
        return f"Error making prediction: {e}"

# 2. Yapay Zeka Ajanını Hazırlama
# IMPORTANT: Replace with your actual Google API Key
db = SQLDatabase.from_uri("sqlite:///nba_database.db")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling", 
    extra_tools=[predict_player_points],
    verbose=False
)

# 3. FastAPI Uygulamasını Başlatma
app = FastAPI()

# HTML/CSS dosyalarını barındıran 'static' klasörünü dışa açıyoruz
app.mount("/static", StaticFiles(directory="static"), name="static")

# Kullanıcıdan gelecek mesajın veri modelini tanımlıyoruz
class ChatRequest(BaseModel):
    message: str

# Ana sayfaya girildiğinde index.html'i göster
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# Frontend'den gelen mesajları yapay zekaya ileten API Uç Noktası (Endpoint)
@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    try:
        # Küçük modellerin (Lite) kaybolmaması için veritabanı yapısını açıkça anlatıyoruz
        database_metadata_hint = (
            "Database Hint: You have access to a table named 'oyuncu_kariyerleri'. "
            "This table contains NBA player statistics. The column 'PLAYER_NAME' contains "
            "the exact text names of the players, such as 'LeBron James', 'Russell Westbrook', and 'Stephen Curry'. "
            "Always query 'oyuncu_kariyerleri' and filter by 'PLAYER_NAME' when asked about a player. "
        )
        
        # Kullanıcının sorusunun başına bu kopyayı, sonuna da netlik talimatını ekliyoruz
        strict_prompt = database_metadata_hint + request.message + " (Answer directly and concisely. Do not explain your thought process or show SQL queries.)"
        
        response = agent_executor.invoke({"input": strict_prompt})
        return {"reply": response["output"]}
    
    except Exception as e:
        print(f"\n" + "="*40 + f"\n!!! BACKEND ERROR !!!\n{str(e)}\n" + "="*40 + "\n")
        raise HTTPException(status_code=500, detail=str(e))
    

    