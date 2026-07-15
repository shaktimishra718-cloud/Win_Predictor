import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from database import get_engine
from team_form import engineer_features

# The exact features your model relies on
FEATURES = ['home_form', 'away_form', 'form_diff', 'fatigue_diff', 'h2h_home_win_rate']
TARGET = 'home_win' # 1 if Home wins, 0 if Draw/Away win

def evaluate_pipeline():
    print("Initiating Model Evaluation...")
    engine = get_engine()
    
    # 1. Load the data
    query = "SELECT * FROM matches ORDER BY match_date ASC"
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df)} historical matches.")

    # 2. Engineer features
    df = engineer_features(df)
    
    # 3. Create the Target Variable (1 if Home Score > Away Score, else 0)
    df[TARGET] = (df['home_score'] > df['away_score']).astype(int)

    # 4. Chronological Train/Test Split (80% Train, 20% Test)
    split_index = int(len(df) * 0.8)
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]

    print(f"\nTraining on {len(train_data)} older matches...")
    print(f"Testing accuracy on {len(test_data)} most recent matches...")

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    # 5. Train a fresh model just for this test
    rf_eval = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        random_state=42, 
        class_weight='balanced'
    )
    rf_eval.fit(X_train, y_train)

    # 6. Predict the Test Set
    predictions = rf_eval.predict(X_test)

    # 7. Print the Report Card
    print("\n" + "="*50)
    print(" 🏆 MODEL ACCURACY REPORT 🏆 ")
    print("="*50)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Detailed Classification Report:")
    # 0 = Home Did Not Win, 1 = Home Won
    print(classification_report(y_test, predictions, target_names=["Draw/Away Win (0)", "Home Win (1)"]))
    
    print("\nConfusion Matrix:")
    matrix = confusion_matrix(y_test, predictions)
    print(f"True Negatives (Correctly predicted Draw/Away): {matrix[0][0]}")
    print(f"False Positives (Predicted Home Win, but they didn't): {matrix[0][1]}")
    print(f"False Negatives (Predicted Draw/Away, but Home Won): {matrix[1][0]}")
    print(f"True Positives (Correctly predicted Home Win): {matrix[1][1]}")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_pipeline()