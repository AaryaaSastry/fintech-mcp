#!/usr/bin/env python
"""
ML-based Fraud Detection Model
Uses Random Forest Classifier trained on transaction data
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load data
DATA_PATH = "data/transactions.csv"
MODEL_PATH = "models/fraud/fraud_model.pkl"
SCALER_PATH = "models/fraud/scaler.pkl"

def load_and_preprocess_data():
    """Load transaction data and engineer features."""
    df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Feature engineering
    features = pd.DataFrame()
    
    # Transaction features
    features['amount'] = df['amount']
    features['amount_log'] = np.log1p(df['amount'])
    
    # Client financial features
    features['credit_score'] = df['credit_score']
    features['yearly_income'] = df['yearly_income']
    features['total_debt'] = df['total_debt']
    features['debt_to_income'] = df['total_debt'] / (df['yearly_income'] + 1)
    features['credit_limit'] = df['credit_limit']
    features['utilization'] = df['credit_limit'] / (df['yearly_income'] + 1)
    
    # Age features
    features['current_age'] = df['current_age']
    features['age_group'] = pd.cut(df['current_age'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5]).astype(int)
    
    # Card features
    features['use_chip'] = df['use_chip'].astype(int)
    features['card_type_encoded'] = df['card_type'].map({'Credit': 1, 'Debit': 0, 'Prepaid': 2}).fillna(0)
    features['card_brand_encoded'] = df['card_brand'].map({'Visa': 0, 'Mastercard': 1, 'Discover': 2, 'Amex': 3}).fillna(0)
    
    # Time features
    features['hour'] = df['timestamp'].dt.hour
    features['day_of_week'] = df['timestamp'].dt.dayofweek
    features['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
    features['is_night'] = ((df['timestamp'].dt.hour >= 22) | (df['timestamp'].dt.hour < 6)).astype(int)
    
    # Merchant features
    features['mcc_code'] = df['mcc_code']
    
    # High-risk indicators
    features['high_amount'] = (df['amount'] > 1000).astype(int)
    features['low_credit_score'] = (df['credit_score'] < 650).astype(int)
    features['high_debt_ratio'] = (features['debt_to_income'] > 0.5).astype(int)
    
    # Labels
    labels = df['is_fraud']
    
    return features, labels

def train_model():
    """Train the fraud detection model."""
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data()
    
    print(f"Data shape: {X.shape}")
    print(f"Fraud cases: {y.sum()} / {len(y)} ({100*y.sum()/len(y):.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',  # Handle imbalanced classes
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_pred = model.predict(X_test_scaled)
    print(f"\nTraining Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model and scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")
    
    return model, scaler

def predict_fraud(transaction_data: dict) -> dict:
    """Predict fraud probability for a single transaction."""
    # Load model and scaler
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training...")
        train_model()
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Prepare features
    df = pd.DataFrame([transaction_data])
    
    features = pd.DataFrame()
    features['amount'] = df['amount']
    features['amount_log'] = np.log1p(df['amount'])
    features['credit_score'] = df['credit_score']
    features['yearly_income'] = df['yearly_income']
    features['total_debt'] = df['total_debt']
    features['debt_to_income'] = df['total_debt'] / (df['yearly_income'] + 1)
    features['credit_limit'] = df['credit_limit']
    features['utilization'] = df['credit_limit'] / (df['yearly_income'] + 1)
    features['current_age'] = df['current_age']
    features['age_group'] = pd.cut(df['current_age'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5]).astype(int)
    features['use_chip'] = df['use_chip']
    features['card_type_encoded'] = df['card_type'].map({'Credit': 1, 'Debit': 0, 'Prepaid': 2}).fillna(0)
    features['card_brand_encoded'] = df['card_brand'].map({'Visa': 0, 'Mastercard': 1, 'Discover': 2, 'Amex': 3}).fillna(0)
    features['hour'] = df['hour']
    features['day_of_week'] = df['day_of_week']
    features['is_weekend'] = df['is_weekend']
    features['is_night'] = df['is_night']
    features['mcc_code'] = df['mcc_code']
    features['high_amount'] = (df['amount'] > 1000).astype(int)
    features['low_credit_score'] = (df['credit_score'] < 650).astype(int)
    features['high_debt_ratio'] = (features['debt_to_income'] > 0.5).astype(int)
    
    # Scale and predict
    X = scaler.transform(features)
    fraud_prob = model.predict_proba(X)[0][1]  # Probability of fraud
    is_fraud = model.predict(X)[0]
    
    return {
        'fraud_probability': round(fraud_prob, 4),
        'is_fraud': bool(is_fraud),
        'risk_level': 'high' if fraud_prob > 0.5 else ('medium' if fraud_prob > 0.2 else 'low')
    }

if __name__ == "__main__":
    # Train the model
    train_model()
