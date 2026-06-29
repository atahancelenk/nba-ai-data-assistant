# 🏀 NBA AI Data Assistant

An end-to-end AI application that answers natural language questions about NBA player statistics and predicts future performance using a custom-trained ML model. Built to demonstrate the full data engineering stack — from raw API ingestion to a deployed agentic AI with tool-calling.

**Live demo:** [https://nba-ai-data-assistant.onrender.com](https://nba-ai-data-assistant.onrender.com)

---

## What it does

Ask the assistant anything about LeBron James, Stephen Curry, or Russell Westbrook:

- *"What season did Westbrook score the most points?"* → queries the SQLite database and returns a formatted stats table
- *"Predict Curry's PPG with 70 GP, 34 MPG, 16 FGA, 5 FTA"* → calls a trained Linear Regression model and returns a prediction
- *"Compare LeBron and Curry career averages side by side"* → the agent decides to use SQL and renders a markdown table in the UI

The agent autonomously decides which tool to use based on the question — no hardcoded routing.

---

## Architecture

```
nba_api  →  etl_pipeline.py  →  SQLite DB
                                    ↓
                           LangChain SQL Agent
                           (llama-3.3-70b via Groq)
                                    ↓
                    ┌───────────────┴───────────────┐
                SQL Tool                    predict_player_points()
           (auto-generated                  (joblib Linear Regression
            by LangChain)                    trained on per-game stats)
                                    ↓
                            FastAPI  /chat
                                    ↓
                        Vanilla HTML/CSS/JS UI
                        (markdown table rendering)
```

**Key design decision:** The agent uses LangChain's tool-calling pattern — the LLM receives tool definitions and decides at runtime whether to run a SQL query, call the ML model, or both. This is more robust than prompt-based routing because the model can chain tools and self-correct failed queries.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data pipeline | Python, nba_api, pandas, SQLAlchemy, SQLite |
| Machine learning | scikit-learn (Linear Regression), joblib |
| AI agent | LangChain, Groq API (llama-3.3-70b-versatile) |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript (vanilla) |
| CI/CD | GitHub Actions |
| Deployment | Render.com |

---

## Engineering decisions worth noting

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

### LLM tool-calling reliability

Smaller LLMs hallucinate SQL table names and column names without guidance. Rather than switching to a larger model, the fix was **context injection** — prepending the table schema and exact column names to every prompt:

```python
database_metadata_hint = (
    "You have access to a table named 'player_careers'. "
    "The column 'PLAYER_NAME' contains values like 'LeBron James', "
    "'Russell Westbrook', and 'Stephen Curry'..."
)
```

The tool docstring also required explicit parameter descriptions with example values — without them, the LLM would mis-map natural language inputs to function arguments and trigger a `Failed to call a function` error.

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

**3. Build the database and train the model**
```bash
python etl_pipeline.py   # pulls data from nba_api, saves to SQLite
python train_model.py    # trains Linear Regression, saves .joblib
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
| SQL | "Compare all three players' career assists side by side" |
| ML | "Predict PPG for a player with 70 GP, 35 MPG, 18 FGA/game, 7 FTA/game" |
| ML | "Predict Curry's points per game next season" |

---

## Known limitations

- **Three players only.** The ML model is trained on LeBron, Curry, and Westbrook — predictions outside this range are extrapolations and should be interpreted cautiously.
- **No API rate limit handling.** Groq's free tier has per-minute limits; sustained load will cause 503 errors that aren't gracefully retried.
- **CI is syntax-only.** The GitHub Actions pipeline checks that `main.py` compiles but does not run integration tests against the agent or database.
- **Static dataset.** `etl_pipeline.py` must be re-run manually to pull updated season data.

---

## Project structure

```
├── etl_pipeline.py        # Pulls data from nba_api, loads into SQLite
├── train_model.py         # Trains and saves the Linear Regression model
├── main.py                # FastAPI app + LangChain agent + ML tool
├── nba_database.db        # SQLite database (generated by ETL)
├── nba_points_predictor.joblib  # Trained model (generated by train_model.py)
├── static/
│   ├── index.html         # Chat UI with markdown table rendering
│   └── style.css          # Dark navy theme
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI
├── requirements.txt
└── .env                   # API keys (not committed)
```