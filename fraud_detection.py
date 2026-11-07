import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Note: This script assumes you have 'train.csv' in the same directory.
# You can download it from the Kaggle Playground S6E11 competition.

def main():
    print("Loading data...")
    try:
        train_df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("Error: Please download train.csv from Kaggle and place it in this folder.")
        return

    # Prepare data
    features = [col for col in train_df.columns if col not in ['id', 'is_fraud'] and train_df[col].dtype != 'object']
    X = train_df[features].fillna(0)
    y = train_df['is_fraud']

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, preds)
    print(f"Validation ROC AUC Score: {score:.5f}")

if __name__ == "__main__":
    main()
