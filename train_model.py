import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def main():
    print("Connecting to the database...")
    engine = create_engine('sqlite:///nba_database.db')
    
    query = "SELECT * FROM player_careers"
    df = pd.read_sql(query, con=engine)
    
    print(f"Successfully loaded {len(df)} rows of player statistics.\n")

    # We want to predict Total Points (PTS) based on Games Played (GP), Minutes (MIN), 
    # Field Goal Attempts (FGA), and Free Throw Attempts (FTA)
    df['MPG'] = df['MIN'] / df['GP']   # Minutes per game
    df['FGA_PG'] = df['FGA'] / df['GP']  # FGA per game
    df['FTA_PG'] = df['FTA'] / df['GP']  # FTA per game
    df['PPG'] = df['PTS'] / df['GP']     # Points per game (target)

    feature_columns = ['GP', 'MPG', 'FGA_PG', 'FTA_PG']
    target_column = 'PPG'
    
    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples\n")

    print("Training the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training completed successfully.\n")

    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("--- Model Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"R-squared (R2) Score: {r2:.4f} (Max is 1.0)")
    print("--------------------------------\n")

    # LOAD / SAVE MODEL
    model_filename = 'nba_points_predictor.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved successfully as '{model_filename}'")

if __name__ == "__main__":
    main()