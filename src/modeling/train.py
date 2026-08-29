import json
import os
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.loading.database import select
from src.modeling.feature_engineering import run_feature_engineering

PARAMS_FILE = "results/xgb_best_params.json"

def get_dataframe() -> pd.DataFrame:
    """Retrieve weather data from database and apply feature engineering.

    Returns:
        pd.DataFrame: Engineered DataFrame ready for model dataset preparation.
    """
    df = select("SELECT * FROM WEATHER")
    df = run_feature_engineering(df=df)
    
    return df

def prepare_data(
    target_col: str ='rain_next_hour',
    time_col: str ='time',
    train_ratio: float = 0.8
    ) -> dict[str, Any]:
    """Prepare train and test splits, feature scaling, and class weight.

    Splits the dataset sequentially without shuffling to preserve time ordering,
    fits a standard scaler on the training set, and computes the ratio of negative
    to positive classes for imbalanced learning.

    Args:
        target_col (str): Column name representing the target label. Defaults
            to 'rain_next_hour'.
        time_col (str): Column name representing the timestamp. Defaults to 'time'.
        train_ratio (float): Fraction of the data to use for training. Defaults
            to 0.8.

    Returns:
        dict[str, Any]: Dictionary containing:
            - 'X_train' (pd.DataFrame): Training feature matrix.
            - 'X_test' (pd.DataFrame): Test feature matrix.
            - 'y_train' (pd.Series): Training target labels.
            - 'y_test' (pd.Series): Test target labels.
            - 'X_train_scaled' (np.ndarray): Standardized training features.
            - 'X_test_scaled' (np.ndarray): Standardized test features.
            - 'scale_pos_weight' (float): Negative to positive class ratio.
    """
    df = get_dataframe()
    
    X = df.drop(columns=[target_col, time_col])
    y = df[target_col]    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scale_pos_weight = (y_train==0).sum() / (y_train == 1).sum()
    
    return{'X_train': X_train, 'X_test': X_test,
           'y_train': y_train, 'y_test': y_test,
           'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
           'scale_pos_weight': scale_pos_weight}
    

def train_models(data: dict[str, Any]) -> dict[str, Any]:
    """Initialize and fit machine learning models on training data.

    Fits Logistic Regression on scaled data, and Random Forest and XGBoost
    on raw training features.

    Args:
        data (dict[str, Any]): Dictionary containing training features, labels,
            scaled datasets, and class weighting.

    Returns:
        dict[str, Any]: Dictionary of trained model instances.
    """
    models = inicialize_models(data['scale_pos_weight'])
    
    models['logistic_regression'].fit(data['X_train_scaled'], data['y_train'])
    
    models['random_forest'].fit(data['X_train'], data['y_train'])
    models['xgboost'].fit(data['X_train'], data['y_train'])
    
    return models

def inicialize_models(scale_pos_weight: float) -> dict[str, Any]:
    """Initialize model instances with default or tuned hyperparameters.

    Loads XGBoost parameters from PARAMS_FILE if present; otherwise, uses
    default parameters.

    Args:
        scale_pos_weight (float): Balancing scale weight for XGBoost positive class.

    Returns:
        dict[str, Any]: Dictionary containing instances of 'logistic_regression',
            'random_forest', and 'xgboost'.
    """
    if os.path.exists(PARAMS_FILE):
        print(f"[INFO] Loading hyperparameters from '{PARAMS_FILE}'...")
        with open(PARAMS_FILE, "r") as f:
            xgb_params = json.load(f)
        xgb = XGBClassifier(**xgb_params)
    else:
        print("[INFO] Optuna parameters  not found. Using default parameters...")
        xgb = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs = -1
        )
        
    models={
        'logistic_regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
            ),
        'random_forest': RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,
            n_jobs=-1,
            random_state=42
            ),
        'xgboost': xgb
    }
    return models