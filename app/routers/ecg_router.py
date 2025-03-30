from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
import numpy as np
import pandas as pd
from ..models.ecg_generator import ECGDataGenerator
from ..models.feature_extractor import ECGFeatureExtractor
from ..models.prediction_model import HeartFailurePredictionModel
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split

router = APIRouter()
ecg_generator = ECGDataGenerator()
feature_extractor = ECGFeatureExtractor()
prediction_model = HeartFailurePredictionModel()

# Store the current abnormality level
current_abnormality_level = 0.5  # Default value

class AbnormalityLevel(BaseModel):
    level: float

@router.post("/set-abnormality-level")
async def set_abnormality_level(abnormality_level: AbnormalityLevel):
    """Set the abnormality level for ECG generation (0.0 to 1.0)"""
    global current_abnormality_level
    if not 0 <= abnormality_level.level <= 1:
        raise HTTPException(status_code=400, detail="Abnormality level must be between 0 and 1")
    current_abnormality_level = abnormality_level.level
    return {"message": f"Abnormality level set to {abnormality_level.level}"}

@router.get("/get-abnormality-level")
async def get_abnormality_level():
    """Get the current abnormality level"""
    return {"abnormality_level": current_abnormality_level}

@router.post("/generate-ecg")
async def generate_ecg(abnormality_type: Optional[str] = None):
    """Generate ECG signals (normal and abnormal)"""
    try:
        # Generate normal ECG
        normal_ecg = ecg_generator.generate_normal_ecg()
        
        # Generate abnormal ECG with current abnormality level
        abnormal_ecg = ecg_generator.generate_abnormal_ecg(abnormality_type)
        
        # Apply abnormality level to the signal
        abnormal_ecg = normal_ecg + (abnormal_ecg - normal_ecg) * current_abnormality_level
        
        # Create time points
        t = np.linspace(0, 10, len(normal_ecg))
        
        return {
            "time": t.tolist(),
            "normal_ecg": normal_ecg.tolist(),
            "abnormal_ecg": abnormal_ecg.tolist(),
            "abnormality_type": abnormality_type or "random",
            "abnormality_level": current_abnormality_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-model")
async def train_model(n_samples: int = 2000, abnormality_ratio: float = 0.4):
    """Train the heart failure prediction model"""
    try:
        print("Generating synthetic dataset...")
        # Generate synthetic dataset
        X, y = ecg_generator.generate_dataset(n_samples, abnormality_ratio)
        
        print("Extracting features...")
        # Extract features
        X_features = []
        for ecg in X:
            features = feature_extractor.extract_features(ecg)
            X_features.append(features)
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X_features)
        
        # Split data: 70% train, 15% validation, 15% test
        train_size = int(0.7 * len(X_df))
        val_size = int(0.15 * len(X_df))
        
        X_train = X_df[:train_size]
        y_train = y[:train_size]
        
        X_val = X_df[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X_df[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Train model with validation data
        model = prediction_model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model on test set
        results = prediction_model.evaluate(X_test, y_test)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict(ecg_data: List[float]):
    """Predict heart failure risk from ECG data"""
    try:
        # Extract features
        features = feature_extractor.extract_features(np.array(ecg_data))
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Make prediction
        prediction = prediction_model.predict(X)[0]
        probability = prediction_model.predict_proba(X)[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "features": features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calculate-heart-failure-risk")
async def calculate_heart_failure_risk(ecg_data: List[float]):
    """Calculate the probability of heart failure based on ECG features"""
    try:
        # Extract features
        features = feature_extractor.extract_features(np.array(ecg_data))
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Get probability
        probability = prediction_model.predict_proba(X)[0][1]
        
        # Calculate risk level
        risk_level = "Low"
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Moderate"
        
        # Get key risk factors
        risk_factors = []
        if features.get('hrv', 0) < 50:
            risk_factors.append("Low Heart Rate Variability")
        if features.get('mean_hr', 0) > 100:
            risk_factors.append("Tachycardia")
        if features.get('mean_hr', 0) < 60:
            risk_factors.append("Bradycardia")
        if features.get('qrs_duration', 0) > 0.12:
            risk_factors.append("Wide QRS Complex")
        
        return {
            "heart_failure_probability": round(probability * 100, 2),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendations": [
                "Schedule a follow-up with a cardiologist",
                "Monitor blood pressure regularly",
                "Maintain a healthy lifestyle",
                "Take prescribed medications as directed"
            ] if risk_level != "Low" else [
                "Continue regular check-ups",
                "Maintain a healthy lifestyle"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/plot-ecg")
async def plot_ecg():
    """Generate and return an ECG plot"""
    # Generate ECG data
    normal_ecg = ecg_generator.generate_normal_ecg()
    abnormal_ecg = ecg_generator.generate_abnormal_ecg()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(ecg_generator.time, normal_ecg)
    plt.title("Normal ECG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(ecg_generator.time, abnormal_ecg)
    plt.title("Abnormal ECG")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.tight_layout()
    
    # Convert plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    
    return plot_base64
