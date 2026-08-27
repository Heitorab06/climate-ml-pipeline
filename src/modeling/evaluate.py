import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
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

def save_metrics(metrics_df: pd.DataFrame, file_path: str = "results/metrics.csv") -> None:
    os.makedirs('results', exist_ok=True)
    metrics_df.to_csv(file_path, index=False)

def load_metrics(file_path: str = "reports/metrics.csv") -> pd.DataFrame:
    return pd.read_csv(file_path)

def generate_graphics(models: dict, data: dict, results: pd.DataFrame, output_dir: str = "results") -> None:
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. PR Curve (All 3 models together)
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        X_test = data['X_test_scaled'] if name == 'logistic_regression' else data['X_test']
        y_proba = model.predict_proba(X_test)[:, 1]
        
        precisions, recalls, _ = precision_recall_curve(data['y_test'], y_proba)
        auc_score = results.loc[results['Model'] == name, 'PR-AUC'].values[0]
        plt.plot(recalls, precisions, label=f"{name} (PR-AUC = {auc_score:.3f})")

    baseline = float(data['y_test'].mean())
    plt.axhline(y=baseline, color='grey', linestyle='--', label=f'Baseline ({baseline:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve comparison')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pr_curve.png", dpi=300)
    plt.close()
    
# 2. Confusion Matrix
    for name, model in models.items():
        X_test = data['X_test_scaled'] if name == 'logistic_regression' else data['X_test']
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(data['y_test'], y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No rain', 'Rain'])
        
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix - {name}')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_{name}.png", dpi=300)
        plt.close()
        
# 3. XGBoost Feature Importance
    if 'xgboost' in models:
        xgb_model = models['xgboost']
        importances = pd.Series(xgb_model.feature_importances_, index=data['X_train'].columns)
        top_10 = importances.nlargest(10).sort_values(ascending=True)

        plt.figure(figsize=(8, 5))
        top_10.plot(kind='barh', color='#1f77b4')
        plt.title('Top 10 Most important features (XGBoost)')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance_xgboost.png", dpi=300)
        plt.close()

    print(f"[INFO] Graphics successefully saved at '{output_dir}/'!")
