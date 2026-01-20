import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle

def train_model(data_path='training_data.csv', model_dir='models'):
    print("Loading training data...")
    if not os.path.exists(data_path):
        print("Data file not found. Run data_generator.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Preprocessing
    # Convert Categorical to One-Hot
    df_encoded = pd.get_dummies(df, columns=['Material', 'Finish'])
    
    # Features X and Targets Y
    # Drop targets and non-feature columns
    X = df_encoded.drop(['Holding_Time', 'Failure_Mode'], axis=1)
    
    y_time = df['Holding_Time']
    y_mode = df['Failure_Mode']
    
    print(f"Features: {X.columns.tolist()}")
    
    # Train Time Predictor (Regressor)
    print("Training Time Predictor (Regressor)...")
    regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    regressor.fit(X, y_time)
    
    # Train Mode Predictor (Classifier)
    print("Training Mode Predictor (Classifier)...")
    classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    classifier.fit(X, y_mode)
    
    # Save Models
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # We save as JSON for compatibility/speed, or pickle for ease of pipeline including columns
    # Let's save the model objects properly
    regressor.save_model(os.path.join(model_dir, 'radar_time_v1.json'))
    classifier.save_model(os.path.join(model_dir, 'radar_mode_v1.json'))
    
    # Also save the feature names to ensure alignment during inference
    with open(os.path.join(model_dir, 'feature_columns.pkl'), 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    print("Models saved successfully in 'models/' directory.")

if __name__ == "__main__":
    train_model()
