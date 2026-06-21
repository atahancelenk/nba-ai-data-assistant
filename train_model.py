import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def main():
    # 1. EXTRACT: Connect to the SQLite database and read the data using SQL
    print("Connecting to the database...")
    engine = create_engine('sqlite:///nba_database.db')
    
    # Query the table we created in the previous step
    query = "SELECT * FROM player_careers"
    df = pd.read_sql(query, con=engine)
    
    print(f"Successfully loaded {len(df)} rows of player statistics.\n")

    # 2. TRANSFORM & FEATURE SELECTION
    # Selecting our features (X) and target variable (y)
    # We want to predict Total Points (PTS) based on Games Played (GP), Minutes (MIN), 
    # Field Goal Attempts (FGA), and Free Throw Attempts (FTA)
    feature_columns = ['GP', 'MIN', 'FGA', 'FTA']
    target_column = 'PTS'
    
    X = df[feature_columns]
    y = df[target_column]

    # Split the dataset into Training set (80%) and Testing set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples\n")

    # 3. MODEL TRAINING
    print("Training the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training completed successfully.\n")

    # 4. MODEL EVALUATION
    # Make predictions on the test set to evaluate performance
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("--- Model Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} points")
    print(f"R-squared (R2) Score: {r2:.4f} (Max is 1.0)")
    print("--------------------------------\n")

    # 5. LOAD / SAVE MODEL
    # Save the trained model to a file so we can reuse it later in our LLM application
    model_filename = 'nba_points_predictor.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved successfully as '{model_filename}'")

if __name__ == "__main__":
    main()