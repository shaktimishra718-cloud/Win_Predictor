import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from database import get_engine
from team_form import engineer_features, FEATURES

def train_and_save_model():
    print("Initiating model training pipeline...")

    # 1. EXTRACT: Load historical data from PostgreSQL
    print("Loading historical data from database...")
    engine = get_engine()
    if not engine:
        raise ConnectionError("Database connection failed.")

    query = "SELECT * FROM matches ORDER BY match_date ASC;"
    df = pd.read_sql(query, engine)

    # 2. TRANSFORM: Pass data through your central feature engine
    print("Applying centralized feature engineering...")
    
    # Flag as historical data so team_form.py knows how to handle it
    df['is_future'] = False 
    
    # Apply your shared logic
    df = engineer_features(df)

    # 3. DEFINE TARGET: What the model is trying to predict
    # Assuming standard binary classification: 1 for Home Win, 0 for Draw/Away
    # Update this if your Colab used a different target calculation
    df['target_home_win'] = (df['home_score'] > df['away_score']).astype(int)

    # 4. PREPARE: Isolate features (X) and target (y)
    X = df[FEATURES]
    y = df['target_home_win']

    # 5. TRAIN: Execute the machine learning algorithm
    print(f"Training RandomForestClassifier on {len(X)} matches...")
    model_temporal = RandomForestClassifier(n_estimators=100, random_state=42)
    model_temporal.fit(X, y)

    # 6. LOAD (Save to disk): Export the trained model as a file
    model_filename = 'model_temporal.joblib'
    joblib.dump(model_temporal, model_filename)
    
    print(f"Success! Model trained and saved locally as '{model_filename}'.")

if __name__ == "__main__":
    train_and_save_model()