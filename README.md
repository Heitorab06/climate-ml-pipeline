# Climate ML Pipeline

An end-to-end data and machine learning pipeline for predicting whether it will rain in the next hour. The project retrieves historical weather observations from the Open-Meteo API, transforms them into modeling-ready data, stores the dataset in PostgreSQL, and trains multiple classification models.

This project demonstrates practical skills in data ingestion, data cleaning, feature engineering, relational data storage, and machine learning experimentation.

## Key Features

- Fetches hourly historical weather data for Salvador, Brazil, using the Open-Meteo API.
- Converts timestamps and prepares a clean Parquet dataset.
- Loads weather data into PostgreSQL through SQLAlchemy.
- Creates lag, rolling, variation, and cyclical time features.
- Defines a binary target for rain in the following hour.
- Trains Logistic Regression, Random Forest, and XGBoost classifiers.
- Uses a chronological train/test split to preserve the time-series order.
- Handles class imbalance with balanced model weights.

## Architecture

```text
Open-Meteo API
      |
      v
Python + Requests
      |
      v
Pandas data cleaning --> Parquet files
      |
      v
PostgreSQL + SQLAlchemy
      |
      v
Feature engineering
      |
      v
Scikit-learn / XGBoost models
```

## Tech Stack

- Python 3.10+
- Requests
- Pandas and NumPy
- PostgreSQL 16
- SQLAlchemy and psycopg2
- Scikit-learn
- XGBoost
- Docker Compose
- Parquet with PyArrow

## Project Structure

```text
.
├── data/                         # Raw and cleaned Parquet datasets
├── src/
│   ├── config.py                 # API parameters and selected weather variables
│   ├── extraction/api.py         # Open-Meteo data extraction
│   ├── transformation/cleaning.py# Timestamp conversion and cleaning
│   ├── loading/database.py       # PostgreSQL connection and data loading
│   └── modeling/
│       ├── feature_engineering.py# Feature and target creation
│       ├── train.py              # Dataset preparation and model training
│       └── evaluate.py           # Reserved for model evaluation
├── docker-compose.yml            # PostgreSQL service
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure PostgreSQL

Create a `.env` file in the project root. The same values are used by Docker Compose and the application:

```env
HOST=localhost
PORT=5432
DB_NAME=weather
DB_USER=postgres
DB_PASSWORD=change-me
```

Start PostgreSQL with Docker Compose:

```bash
docker compose up -d postgres
```

### 4. Run the pipeline stages

Run these commands from the project root:

```bash
python -c "from src.extraction.api import get_weather_data; get_weather_data().to_parquet('data/weather_raw.parquet', index=False)"
python -c "import pandas as pd; from src.transformation.cleaning import clean_data; clean_data(pd.read_parquet('data/weather_raw.parquet')).to_parquet('data/weather_clean.parquet')"
python -c "from src.loading.database import upload_to_postgres; upload_to_postgres()"
python -m src.modeling.train
```

The extraction, cleaning, loading, and feature engineering functions are also available for reuse in notebooks or orchestration code. The expected intermediate files are `data/weather_raw.parquet` and `data/weather_clean.parquet`.

## Modeling Approach

The target variable, `rain_next_hour`, is set to `1` when the next hour's precipitation is at least `0.02`. The feature engineering stage includes:

- 1-, 2-, 3-, and 6-hour lag features;
- three-hour precipitation accumulation;
- three-hour variation for selected weather variables;
- cyclical hour-of-day and day-of-year features.

The dataset is split chronologically, with 80% used for training and 20% for testing. Logistic Regression receives standardized features, while Random Forest and XGBoost use the original feature scale.

## Current Status

The ingestion, transformation, database loading, feature engineering, and model training stages are implemented. Model evaluation, workflow orchestration with Apache Airflow, and serving predictions through FastAPI are planned next steps.

## Roadmap

- Add evaluation metrics such as precision, recall, F1-score, and ROC-AUC.
- Compare model performance and persist the best model.
- Add automated orchestration with Apache Airflow.
- Expose predictions through a FastAPI service.
- Add automated tests, data validation, and CI/CD.

## License

This project is intended for educational and portfolio purposes.
