import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
from db import engine

FEATURE_COLUMNS = ['GP', 'MPG', 'FGA_PG', 'FTA_PG']

# Each target gets its own model + its own saved file.
TARGETS = {
    'PPG': 'nba_points_predictor.joblib',
    'RPG': 'nba_rebounds_predictor.joblib',
    'APG': 'nba_assists_predictor.joblib',
}

CV_FOLDS = 5

def load_data():
    print("Connecting to the database...")
    df = pd.read_sql("SELECT * FROM player_careers", con=engine)
    print(f"Successfully loaded {len(df)} rows of player statistics.\n")

    # Guard against divide-by-zero for any 0-GP rows (e.g. injured/DNP seasons)
    df = df[df['GP'] > 0].copy()

    df['MPG'] = df['MIN'] / df['GP']
    df['FGA_PG'] = df['FGA'] / df['GP']
    df['FTA_PG'] = df['FTA'] / df['GP']
    df['PPG'] = df['PTS'] / df['GP']
    df['RPG'] = df['REB'] / df['GP']
    df['APG'] = df['AST'] / df['GP']

    return df

def evaluate_candidates(X, y):
    """
    Runs K-fold cross-validation for each candidate model and returns
    per-model mean/std MAE and mean R2. A single train/test split can make
    a model look better or worse than it really is just from the luck of
    the split; cross-validation averages over several splits instead.
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    candidates = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42
        ),
    }

    results = {}
    for name, model in candidates.items():
        mae_scores = -cross_val_score(
            model, X, y, cv=kf, scoring='neg_mean_absolute_error'
        )
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        results[name] = {
            'model': model,
            'mean_mae': mae_scores.mean(),
            'std_mae': mae_scores.std(),
            'mean_r2': r2_scores.mean(),
        }
    return results

def train_and_save_target(df, target_col, output_filename):
    X = df[FEATURE_COLUMNS]
    y = df[target_col]

    print(f"=== Target: {target_col} ===")
    print(f"{CV_FOLDS}-fold cross-validation on {len(X)} rows\n")

    results = evaluate_candidates(X, y)

    for name, res in results.items():
        print(
            f"  {name:<18s} MAE: {res['mean_mae']:.3f} "
            f"(+/- {res['std_mae']:.3f})   R2: {res['mean_r2']:.4f}"
        )

    best_name = min(results, key=lambda n: results[n]['mean_mae'])
    best_model = results[best_name]['model']
    print(f"\n  Winner: {best_name} (lowest cross-validated MAE)")

    # Cross-validation is only used to pick the model. Refit the winner on
    # the FULL dataset before saving, so the deployed model uses all
    # available data rather than just one CV fold's worth.
    best_model.fit(X, y)

    joblib.dump(
        {
            'model': best_model,
            'model_name': best_name,
            'features': FEATURE_COLUMNS,
            'target': target_col,
        },
        output_filename,
    )
    print(f"  Saved -> '{output_filename}'\n")

    return best_name, results[best_name]['mean_mae'], results[best_name]['mean_r2']

def main():
    df = load_data()

    summary = []
    for target_col, output_filename in TARGETS.items():
        best_name, mae, r2 = train_and_save_target(df, target_col, output_filename)
        summary.append((target_col, best_name, mae, r2))

    print("--- Summary ---")
    for target_col, best_name, mae, r2 in summary:
        print(f"{target_col}: {best_name:<18s} MAE={mae:.3f}  R2={r2:.4f}")

if __name__ == "__main__":
    main()