from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
from src.modeling.train import prepare_model_dataset
from src.modeling.train import train_models

from sklearn.metrics import(
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
    )

def evaluate_models(models: dict[str, Any], data: dict[str, Any])-> dict[str, Any]:
    results = []

    for name, model in models.items():
        X_test = data['X_test_scaled'] if name == "logistic_regression" else data['X_test']

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            'Model': name,
            'PR-AUC': average_precision_score(data["y_test"], y_proba),
            'F1_Score': f1_score(data['y_test'], y_proba),
            'Recall': recall_score(data['y_test'], y_pred),
            'Precision': precision_score(data['y_test'], y_pred)
        })

        return pd.DataFrame(results).sort_values(by='PR-AUC', ascending=False)
