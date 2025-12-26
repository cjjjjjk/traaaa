from flask import Blueprint, jsonify
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

create_model_bp = Blueprint('create_model', __name__)

@create_model_bp.route('/train-logistic', methods=['POST', 'GET'])
def train_logistic_regression():
    """
    train logistic regression model on entire dataset from database.csv
    save model and scaler to app/utils/model/logicstic/
    """
    try:
        # determine paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        # path to database.csv
        data_path = os.path.join(app_dir, 'data', 'raw', 'databases.csv')
        
        # path to save model
        model_dir = os.path.join(app_dir, 'utils', 'model', 'logicstic')
        os.makedirs(model_dir, exist_ok=True)
        
        # check if database.csv exists
        if not os.path.exists(data_path):
            return jsonify({
                "status": "error",
                "message": f"database.csv not found at {data_path}"
            }), 404
        
        # 1. load data
        df = pd.read_csv(data_path)
        
        # convert comma to dot for float columns
        cols_to_convert = ['vectors_chao_score', 'congestion_score']
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)
        
        # 2. define features (X) and target (Y)
        feature_cols = ['car_count', 'truck_count', 'bus_count', 'motorcycle_count', 'road_area_pixels', 'vectors_chao_score']
        
        # check if all required columns exist
        missing_cols = [col for col in feature_cols + ['congestion_score'] if col not in df.columns]
        if missing_cols:
            return jsonify({
                "status": "error",
                "message": f"missing columns in database.csv: {missing_cols}"
            }), 400
        
        X = df[feature_cols]
        Y_continuous = df['congestion_score']
        
        # convert target to binary (threshold 0.38)
        threshold = 0.45
        Y_binary = (Y_continuous >= threshold).astype(int)
        
        # 3. normalize features (min-max scaling)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 4. train logistic regression model on entire dataset
        log_model = LogisticRegression(random_state=42)
        log_model.fit(X_scaled, Y_binary)
        
        # 5. save model and scaler
        model_path = os.path.join(model_dir, 'logistic_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(log_model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # get training statistics
        train_accuracy = log_model.score(X_scaled, Y_binary)
        positive_samples = int(Y_binary.sum())
        negative_samples = int(len(Y_binary) - Y_binary.sum())
        
        return jsonify({
            "status": "success",
            "message": "logistic regression model trained successfully on databases.csv",
            "details": {
                "total_samples": len(df),
                "positive_samples": positive_samples,
                "negative_samples": negative_samples,
                "threshold": threshold,
                "train_accuracy": float(train_accuracy),
                "features": feature_cols,
                "model_path": model_path,
                "scaler_path": scaler_path
            }
        }), 200
        
    except Exception as e:
        print(f"[ERROR] train_logistic_regression: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
