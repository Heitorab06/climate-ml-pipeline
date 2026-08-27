import os
from src.modeling.train import prepare_data, train_models
from src.modeling.evaluate import evaluate_models, save_metrics, load_metrics, generate_graphics
from dotenv import load_dotenv

load_dotenv()
METRICS_FILE = os.getenv("METRICS_FILE")

def run_pipeline():
    print("1. Preparing data...")
    data = prepare_data()

    print("2. Training models...")
    models = train_models(data=data)

    print("3. Generating test metrics")
    results = evaluate_models(models, data)

    print("4. Saving metrics")
    save_metrics(results, METRICS_FILE)

    print(results.head())
    return results


if __name__ == "__main__":
    run_pipeline()