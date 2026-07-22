# 🏀 NBA AI Data Assistant

An end-to-end AI application that answers natural language questions about NBA player statistics, tracks a custom Elo-based skill rating, and predicts future performance using a self-trained ML model. Built to demonstrate the full data engineering stack — from raw API ingestion to a deployed agentic AI with tool-calling.

**Live demo:** [https://nba-ai-data-assistant.onrender.com](https://nba-ai-data-assistant.onrender.com)

---

## What it does

Ask the assistant natural language questions about any active NBA player:

- *"What season did Westbrook score the most points?"* → queries the database and returns a formatted stats table
- *"Predict Curry's PPG with 70 GP, 34 MPG, 16 FGA, 5 FTA"* → calls a trained Linear Regression model and returns a prediction
- *"Show me the Elo rating leaderboard"* → returns ranked Elo ratings computed from real head-to-head games
- *"Compare LeBron and Curry career averages side by side"* → the agent decides to use SQL and renders a markdown table in the UI

The agent autonomously decides which tool to use based on the question — no hardcoded routing.

---

## Architecture

```
nba_api  →  etl_pipeline.py  →  Postgres DB via db.py (player_careers, ~500+ active players)
                                    (SQLite fallback for local dev when DATABASE_URL is unset)
                                    │
                        build_elo_ratings.py  →  player_game_logs, player_elo, player_elo_history
                                    ↓
                           LangChain SQL Agent
                           (llama-3.3-70b via Groq)
                                    ↓
              ┌─────────────────────┼─────────────────────┐
          SQL Tool          predict_player_points()   get_elo_leaderboard()
     (auto-generated         (joblib Linear Regression   (queries player_elo,
      by LangChain)           trained on per-game stats)  ranked by rating)
                                    ↓
                            FastAPI  /chat, /players
                                    ↓
                        Vanilla HTML/CSS/JS UI
                        (markdown table rendering)
```

**Key design decision:** The agent uses LangChain's tool-calling pattern — the LLM receives tool definitions and decides at runtime whether to run a SQL query, call the ML model, call the Elo leaderboard tool, or chain several of these. This is more robust than prompt-based routing because the model can chain tools and self-correct failed queries.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data pipeline | Python, nba_api, pandas, SQLAlchemy, Postgres (SQLite fallback for local dev) |
| Machine learning | scikit-learn (Linear Regression), joblib |
| Rating system | Custom Elo implementation (K=20), Hollinger's Game Score |
| AI agent | LangChain, Groq API (llama-3.3-70b-versatile) |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript (vanilla) |
| CI/CD | GitHub Actions |
| Deployment | Render.com |

---

## Engineering decisions worth noting

### Scaling the dataset from 3 players to the full active roster

The dataset originally covered only 3 hardcoded players. `etl_pipeline.py` now pulls every active NBA player via `players.get_active_players()` — roughly 500+ players — into `player_careers`, with resume support (skips players already saved, in case the run is interrupted) and a retry-with-backoff loop for flaky API calls.

The motivation wasn't stability (the app never crashed with 3 players) — it was **model generalization and portfolio credibility**. A regression model trained on 3 players' careers can't meaningfully generalize; training on hundreds of players' career arcs gives the model an actual distribution to learn from, and reviewers evaluating the project don't have to wonder why the "dataset" is 3 named individuals.

### Per-game normalization in the ML model

The first version of the model trained on **season totals** (e.g. `MIN = 2,800`, `PTS = 2,250`) but the prediction tool accepted **per-game averages** (e.g. `MIN = 35`). The model had never seen inputs that small, so it extrapolated wildly — returning predictions like `-424 points`.

The fix was to normalize training data to per-game features before fitting:

```python
df['MPG']    = df['MIN'] / df['GP']
df['FGA_PG'] = df['FGA'] / df['GP']
df['FTA_PG'] = df['FTA'] / df['GP']
df['PPG']    = df['PTS'] / df['GP']   # new target
```

This is a concrete lesson in **training/inference data consistency** — the features at prediction time must match the scale the model learned from.

### Building a legitimate Elo rating system

Rather than fabricate a rating from season averages, `build_elo_ratings.py` derives ratings from **real head-to-head games**: it pulls per-game logs, self-joins on `GAME_ID` to find games where two tracked players actually played, scores each player's performance with **Hollinger's Game Score** (a single-number box-score metric), and runs a chronological Elo update (K=20, starting rating 1500) across the results.

This produces three tables: `player_game_logs` (raw per-game stats), `player_elo` (current rating + games played per player), and `player_elo_history` (every rating change, game by game — useful for trend questions).

### LLM tool-calling reliability

Smaller LLMs hallucinate SQL table names and column names without guidance. Rather than switching to a larger model, the fix was **context injection** — prepending table schema and column descriptions to every prompt, so the LLM knows about `player_careers`, `player_elo`, and `player_elo_history` and how they relate.

The tool docstrings also require explicit parameter descriptions with example values — without them, the LLM mis-maps natural language inputs to function arguments and triggers a `Failed to call a function` error.

### Silent frontend failures

The original error response from FastAPI had a shape mismatch — the frontend expected `data.reply` but error responses returned `data.detail`. Failed requests rendered nothing, making it look like the app hung. Adding explicit error shape handling in the JS made failures visible rather than silent.

---

## Running locally

**1. Clone and install**
```bash
git clone https://github.com/atahancelenk/nba-ai-data-assistant.git
cd nba-ai-data-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your Groq API key**
```bash
# .env
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com).

**Optional: point at Postgres instead of local SQLite**
```bash
# .env
DATABASE_URL=postgresql://user:password@host:port/dbname
```
`db.py` reads `DATABASE_URL` and falls back to a local `nba_database.db` SQLite file if it's unset — handy for quick local runs, but for a real refresh of the production data, set this to your Render Postgres instance's **external** connection string (the internal one only resolves inside Render's network).

**3. Build the database, Elo ratings, and train the model**
```bash
python etl_pipeline.py        # pulls all active players from nba_api into the database (long-running)
python build_elo_ratings.py   # derives Elo ratings from real head-to-head games
python train_model.py         # trains Linear Regression, saves .joblib
```

**4. Start the server**
```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

---

## Example prompts

| Type | Prompt |
|---|---|
| SQL | "What is LeBron's career scoring average?" |
| SQL | "What season did Westbrook score the most points?" |
| SQL | "Compare LeBron and Curry's career assists side by side" |
| ML | "Predict PPG for a player with 70 GP, 35 MPG, 18 FGA/game, 7 FTA/game" |
| Elo | "Show me the Elo rating leaderboard" |
| Elo | "How has Curry's Elo rating changed over time?" |

---

## Known limitations

- **Game Score is a proxy metric, not a true win/loss outcome.** Since players on different teams rarely "compete" in a scoreboard sense, head-to-head results are approximated by comparing individual performance (Game Score) in games where both happened to play — a deliberate simplification, not a measure of team victory.
- **K=20 is a fixed constant**, not tuned against a validation set — a reasonable default borrowed from chess Elo conventions, not empirically optimized for basketball.
- **No API rate limit handling.** Groq's free tier has per-minute limits; sustained load will cause 503 errors that aren't gracefully retried.
- **CI is syntax-only.** The GitHub Actions pipeline checks that `main.py` compiles but does not run integration tests against the agent or database.
- **Static dataset.** `etl_pipeline.py` and `build_elo_ratings.py` must be re-run manually to pull updated season/game data.

---

## Project structure

```
├── etl_pipeline.py              # Pulls all active players from nba_api, loads into SQLite
├── build_elo_ratings.py         # Derives Elo ratings from real head-to-head game logs
├── train_model.py               # Trains and saves the Linear Regression model
├── main.py                      # FastAPI app + LangChain agent + ML/Elo tools
├── db.py                        # Central DB connection — Postgres via DATABASE_URL, SQLite fallback locally
├── nba_database.db              # Local SQLite fallback (only generated if DATABASE_URL is unset)
├── nba_points_predictor.joblib  # Trained model (generated by train_model.py)
├── static/
│   ├── index.html               # Chat UI with markdown table rendering
│   └── style.css                # Dark navy theme
├── .github/workflows/
│   └── ci.yml                   # GitHub Actions CI
├── requirements.txt
└── .env                         # API keys (not committed)
```