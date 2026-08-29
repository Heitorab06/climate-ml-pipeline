import json
import os

import numpy as np
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

PARAMS_PATH = "results/xgb_best_params.json"

def run_tuning(data: dict) -> dict:
    """Optimize XGBoost hyperparameters using Optuna and TimeSeriesSplit.

    Executes a 50-trial Bayesian optimization study to maximize PR-AUC using
    5-fold time-series cross-validation. The best hyperparameters are saved to
    PARAMS_PATH as a JSON file.

    Args:
        data (dict): Dictionary containing the dataset splits, specifically
            'X_train' and 'y_train'.

    Returns:
        dict: Dictionary containing the best hyperparameters found by Optuna.
    """
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 15.0),
            'random_state': 42,
            'early_stopping_rounds': 50,
            'eval_metric': 'aucpr',
            'n_jobs': -1
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(data['X_train']):
            
            X_train, X_val = data['X_train'].iloc[train_idx], data['X_train'].iloc[val_idx]
            y_train, y_val = data['y_train'].iloc[train_idx], data['y_train'].iloc[val_idx]
            
            model = XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_proba = model.predict_proba(X_val)[:, 1]
            
            score = average_precision_score(y_val, y_proba)
            cv_scores.append(score)
            
        return np.mean(cv_scores)
    
    print("[OPTUNA] Inicializing optimization with 50 trials")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print(f"Best PR-AUC in validation: {study.best_value:.4f}")
    best_params = study.best_params
    
    with open(PARAMS_PATH, 'w') as f:
        json.dump(best_params, f, indent=4)
    return best_params


