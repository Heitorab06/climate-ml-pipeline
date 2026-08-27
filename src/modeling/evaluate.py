from typing import Any
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.metrics import(
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
    )

def evaluate_models(models: dict[str, Any], data: dict[str, Any])-> pd.DataFrame:
    results = []

    for name, model in models.items():
        X_test = data['X_test_scaled'] if name == "logistic_regression" else data['X_test']

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            'Model': name,
            'PR-AUC': average_precision_score(data["y_test"], y_proba),
            'F1_Score': f1_score(data['y_test'], y_pred),
            'Recall': recall_score(data['y_test'], y_pred),
            'Precision': precision_score(data['y_test'], y_pred)
        })

    results_df = pd.DataFrame(results).sort_values(by='PR-AUC', ascending=False)


    return results_df

def save_metrics(metrics_df: pd.DataFrame, file_path: str = "reports/metrics.csv") -> None:
    metrics_df.to_csv(file_path, index=False)

def load_metrics(file_path: str = "reports/metrics.csv") -> pd.DataFrame:
    return pd.read_csv(file_path)

def generate_graphics(metrics: pd.DataFrame) -> None:
    
    pass
