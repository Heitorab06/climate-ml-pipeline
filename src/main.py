import argparse
import os

from src.extraction.api import get_weather_data
from src.loading.database import check_database_has_data, save_to_database
from src.modeling.evaluate import (
    evaluate_models,
    load_metrics,
    save_metrics,
    generate_graphics,
)
from src.modeling.train import prepare_data, train_models
from src.transformation.cleaning import clean_data
from src import config

METRICS_FILE = config.METRICS_FILE

def run_data_pipeline() -> None:
    """Execute data pipeline: Extraction -> Cleaning -> Upload to database"""
    
    print("\n" + "="*40)
    print(">>> EXECUTING DATA PIPELINE (ETL)")
    print("="*40)
    
    print("[ETL 1/3] Extracting raw data from API...")
    raw_file = get_weather_data()
    
    print("[ETL 2/3] Cleaning and formatting data...")
    clean_file = clean_data(raw_file)
    
    print("[ETL 3/3] Loading new data do database...")
    save_to_database(clean_file)
    print("Data Pipeliene successfully completed")
    
    
def run_ml_pipeline():
    print("1. Preparing data...")
    data = prepare_data()

    print("2. Training models...")
    models = train_models(data=data)

    print("3. Generating test metrics...")
    results = evaluate_models(models, data)

    print("4. Saving metrics...")
    save_metrics(results, METRICS_FILE)
    generate_graphics(models, data, results)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="End-to-End weather prediction pipeline")
    parser.add_argument(
        "--etl", 
        action="store_true", 
        help="Execute API extraction and load to database before training"
    )
    parser.add_argument(
        "--retrain", 
        action="store_true", 
        help="Force retrain of the models."
    )
    args = parser.parse_args()

    if args.etl or not check_database_has_data():
        run_data_pipeline()
    else:
        print("[INFO] Database ready. Skipping API data extraction (use --etl to force it).")
        
    if args.retrain or not os.path.exists(METRICS_FILE):
        if args.retrain:
            print("[INFO] Flag --retrain detected. Reexecuting training...")
        else:
            print("[INFO] Metrics not found (first execution).")
            
        metrics_df = run_ml_pipeline()
    else:
        print(f"[INFO] Loading saved metrics from '{METRICS_FILE}' (add --retrain to train again):")
        metrics_df = load_metrics(METRICS_FILE)

    print("\n=== Model Performance ===")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()