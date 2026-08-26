from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.loading.database import select
from src.modeling.feature_engineering import run_feature_engineering


def get_dataframe() -> pd.DataFrame:
    df = select("SELECT * FROM WEATHER")
    df = run_feature_engineering(df=df)
    
    return df

def prepare_model_dataset(
    target_col: str ='rain_next_hour',
    time_col: str ='time',
    train_ratio: float = 0.8
    ) -> dict[str, Any]:
    
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
    
    models = inicialize_models(data['scale_pos_weight'])
    
    models['logistic_regression'].fit(data['X_train_scaled'], data['y_train'])
    
    models['random_forest'].fit(data['X_train'], data['y_train'])
    models['xgboost'].fit(data['X_train'], data['y_train'])
    
    return models

def inicialize_models(scale_pos_weight: float) -> dict[str, Any]:
    
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
        'xgboost': XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=42
                )
    }
    return models

data = prepare_model_dataset()
models = train_models(data=data)




    
