from src.modeling.train import prepare_data, train_models
from src.modeling.evaluate import evaluate_models

def run_pipeline():
    data = prepare_data("data/climate.csv")
    
    models = train_models(data=data)
    
    results = evaluate_models(models, data)
    print(results)

if __name__ == "__main__":
    run_pipeline()