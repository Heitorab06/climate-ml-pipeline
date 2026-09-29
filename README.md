# Climate ML Pipeline

An end-to-end data and machine learning pipeline for predicting whether it will rain in the next hour. The project retrieves historical hourly weather observations from the Open-Meteo API, transforms and loads the dataset into PostgreSQL, engineers time-series features, and trains multiple classification models to evaluate their predictive performance.

This project demonstrates practical skills in data ingestion, ETL pipelining, relational data storage, time-series feature engineering, machine learning modeling, and performance evaluation for imbalanced data.

---

## Dataset & Class Imbalance

The pipeline processes historical hourly meteorological observations for Salvador, Brazil. A positive target (`rain_next_hour = 1`) is defined when precipitation in the subsequent hour is $\ge 0.2\text{ mm}$.

The dataset exhibits an inherent class imbalance between non-rain and rain periods:

```text
rain_next_hour
0    20536
1     5762
```

| Class | Definition | Samples | Percentage |
| :---: | :--- | :---: | :---: |
| `0` | No rain (`precipitation < 0.2 mm`) | 20,536 | 78.09% |
| `1` | Rain (`precipitation >= 0.2 mm`) | 5,762 | 21.91% |
| **Total** | | **26,298** | **100.0%** |

This ~3.56:1 negative-to-positive ratio makes raw accuracy misleading. The modeling strategy directly addresses this asymmetry through cost-sensitive learning (`scale_pos_weight` and balanced class weights) and prioritizes **PR-AUC (Precision-Recall AUC)** and recall to avoid missing rain events.

---

## Key Features

- **Automated Ingestion (ETL):** Fetches hourly historical weather data for Salvador, Brazil from the Open-Meteo API and stores cleaned records directly into PostgreSQL.
- **In-Memory Transformation:** Streamlined data processing without intermediate file dependencies.
- **Time-Series Feature Engineering:** Generates lag features (1h, 2h, 3h, 6h), rolling precipitation sums (3h), variable variations (3h), and cyclical temporal features (sin/cos for hour of day and day of year).
- **Binary Target Definition:** Predicts rainfall in the following hour (`precipitation >= 0.2 mm`).
- **Classification Models:** Trains and compares Logistic Regression, Random Forest, and XGBoost classifiers.
- **Hyperparameter Optimization:** Bayesian Optimization with Optuna (50 trials using `TimeSeriesSplit`) to fine-tune XGBoost hyperparameters.
- **Chronological Split & Class Imbalance:** Uses chronological train/test split (80/20) to prevent temporal data leakage and handles class imbalance with `scale_pos_weight` and balanced class weights.
- **Evaluation & Visual Artifacts:** Generates Precision-Recall curves, confusion matrices, feature importance charts, and tabular metrics (PR-AUC, F1-Score, Recall, Precision).
- **CLI Orchestration:** Unified entry point via `src.main` with flexible flags for ETL, model retraining, and Optuna hyperparameter tuning.

---

## Architecture

```text
┌─────────────────────────┐
│     Open-Meteo API      │
└────────────┬────────────┘
             │ (Requests)
             ▼
┌─────────────────────────┐
│    Pandas Data ETL      │
│  (In-Memory Cleaning)   │
└────────────┬────────────┘
             │ (SQLAlchemy)
             ▼
┌─────────────────────────┐
│   PostgreSQL Database   │
└────────────┬────────────┘
             │ (SQL Query)
             ▼
┌─────────────────────────┐
│   Feature Engineering   │
│  (Lags, Rolling, Sins)  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Model Training & Eval   │
│ (LR, Random Forest, XGB)│
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Results & Metrics    │
│ (PR Curve, CM, CSV)     │
└─────────────────────────┘
```

---

## Tech Stack

- **Language:** Python 3.10+
- **Data Manipulation & Extraction:** Pandas, NumPy, Requests
- **Database & ORM:** PostgreSQL 16, SQLAlchemy, Psycopg2
- **Machine Learning & Optimization:** Scikit-learn, XGBoost, Optuna
- **Data Visualization:** Matplotlib
- **Infrastructure:** Docker & Docker Compose

---

## Project Structure

```text
.
├── docker-compose.yml            # PostgreSQL container definition
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
├── results/                      # Generated evaluation metrics and charts
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_xgboost.png
│   ├── feature_importance_xgboost.png
│   ├── metrics.csv
│   ├── pr_curve.png
│   └── xgb_best_params.json      # Optimal hyperparameters found by Optuna
└── src/
    ├── config.py                 # API parameters, coordinates, and file paths
    ├── main.py                   # Main pipeline orchestrator and CLI entrypoint
    ├── extraction/
    │   └── api.py                # Open-Meteo data extraction
    ├── transformation/
    │   └── cleaning.py           # Timestamp formatting and data cleaning
    ├── loading/
    │   └── database.py           # PostgreSQL connection, loading, and queries
    └── modeling/
        ├── feature_engineering.py# Lags, rolling metrics, cyclical features & target
        ├── train.py              # Data prep, scaling, and model training
        ├── tune.py               # Bayesian optimization with Optuna (50 trials)
        └── evaluate.py           # Evaluation metrics and chart generation
```

---

## Getting Started

### 1. Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

On Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory (used by Docker Compose and the Python application):

```env
HOST=localhost
PORT=5432
DB_NAME=weather
DB_USER=postgres
DB_PASSWORD=change-me
```

### 4. Start the Database

Launch the PostgreSQL service with Docker Compose:

```bash
docker compose up -d postgres
```

---

## Running the Pipeline

The entire pipeline is orchestrated through `src.main`.

### Default Run

```bash
python -m src.main
```

- If the database is empty, it automatically executes the ETL pipeline (fetches data from the API, cleans it, and loads it into PostgreSQL).
- If metrics already exist in `results/metrics.csv`, it displays the cached performance table. If not, it trains all models, generates evaluation plots, and saves the metrics.

### CLI Options

| Flag | Description | Example |
| :--- | :--- | :--- |
| *(None)* | Default execution. Uses database cache if populated and loads saved metrics if available. | `python -m src.main` |
| `--etl` | Forces re-extraction from Open-Meteo API, cleans the data, and updates the PostgreSQL database. | `python -m src.main --etl` |
| `--retrain` | Forces model retraining, re-evaluates all classifiers, and updates plots in `results/`. | `python -m src.main --retrain` |
| `--tune` | Runs Bayesian Optimization (Optuna) with 50 trials across 5-fold `TimeSeriesSplit` cross-validation to search for optimal XGBoost hyperparameters, saves them to `results/xgb_best_params.json`, and retrains all models. *(Note: May take a few minutes to execute due to 50 trials).* | `python -m src.main --tune` |
| `--etl --retrain` | Runs the complete end-to-end pipeline from scratch (API extraction $\rightarrow$ DB $\rightarrow$ Training $\rightarrow$ Evaluation). | `python -m src.main --etl --retrain` |
| `--etl --tune` | Full run: re-extracts data, updates DB, executes Optuna hyperparameter tuning (50 trials), and retrains all models. | `python -m src.main --etl --tune` |

---

## Modeling Approach

### 1. Target Definition
The target variable `rain_next_hour` is binary, defining rainfall as hourly precipitation $\ge 0.2\text{ mm}$ in the next hour:
$$\text{rain\_next\_hour} = \begin{cases} 1, & \text{if precipitation in the next hour } \ge 0.2\text{ mm} \\ 0, & \text{otherwise} \end{cases}$$

Across the entire dataset of 26,298 hours:
- **Class `0` (No Rain):** 20,536 observations (78.09%)
- **Class `1` (Rain):** 5,762 observations (21.91%)

### 2. Feature Engineering
- **Lag Features:** 1, 2, 3, and 6-hour lags for `temperature_2m`, `relative_humidity_2m`, `pressure_msl`, `wind_speed_10m`, and `precipitation`.
- **Rolling Aggregations:** 3-hour cumulative precipitation (`cumulative_precipitation_3h`).
- **Variations:** 3-hour delta for temperature, humidity, pressure, and wind speed.
- **Cyclical Temporal Features:** Sine and cosine transformations for hour of day (`hour_sin`, `hour_cos`) and day of year (`day_sin`, `day_cos`).

### 3. Model Training & Evaluation
- **Chronological Split:** 80% train / 20% test without shuffling to preserve temporal integrity.
- **Feature Scaling:** `StandardScaler` applied to Logistic Regression; tree-based models (Random Forest, XGBoost) use raw features.
- **Class Imbalance:** Handled via `class_weight='balanced'` (Logistic Regression, Random Forest) and `scale_pos_weight` (XGBoost).
- **Evaluation Metrics:** PR-AUC (Average Precision), F1-Score, Recall, Precision, and Confusion Matrices saved to `results/`.

### 4. Hyperparameter Tuning (Optuna)
- **Model Selected:** XGBoost (demonstrated the strongest baseline predictive capability).
- **Optimization Algorithm:** Bayesian Optimization via Optuna (Tree-structured Parzen Estimator - TPE).
- **Validation Scheme:** 5-fold `TimeSeriesSplit` cross-validation on the training set to prevent temporal lookahead leakage.
- **Optimization Metric:** Maximizes PR-AUC (Average Precision score) with early stopping (`early_stopping_rounds=50`).
- **Search Space (50 Trials):**
  - `n_estimators`: [50, 300]
  - `max_depth`: [2, 10]
  - `learning_rate`: [0.01, 0.3] (log scale)
  - `subsample`: [0.6, 1.0]
  - `scale_pos_weight`: [5.0, 15.0]
- **Persistence & Best Parameters:** Best hyperparameter configuration is automatically saved to `results/xgb_best_params.json` and reused during subsequent retraining:
  - `n_estimators`: `194`
  - `max_depth`: `4`
  - `learning_rate`: `0.047747884330947565`
  - `subsample`: `0.9761788582619335`
  - `scale_pos_weight`: `7.2839045017915405`

---

## Model Performance & Tuning Comparison

The models were evaluated on the chronological 20% holdout test set. Given the imbalanced nature of rainfall occurrences, **PR-AUC (Precision-Recall Area Under Curve)** serves as the primary benchmark metric.

### Baseline Performance (Before Tuning)

| Model | PR-AUC | F1-Score | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Baseline)** | **0.721579** | 0.653892 | 0.743869 | 0.583333 |
| **Random Forest** | 0.718712 | **0.660262** | 0.686649 | **0.635828** |
| **Logistic Regression** | 0.712911 | 0.647917 | **0.847411** | 0.524452 |

### Tuned Performance (After Optuna Optimization — 50 Trials)

| Model | PR-AUC | F1-Score | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **0.748501** *(+0.0269)* | 0.643219 | **0.921889** *(+0.1780)* | 0.493917 |
| **Random Forest** | 0.718712 | **0.660262** | 0.686649 | **0.635828** |
| **Logistic Regression** | 0.712911 | 0.647917 | 0.847411 | 0.524452 |

### XGBoost Hyperparameters: Baseline vs. Tuned

The baseline model used default XGBoost parameters, with `scale_pos_weight` set dynamically to the training set class imbalance ratio ($N_{\text{negative}} / N_{\text{positive}} \approx 3.5136$). The tuned configuration was obtained through Bayesian Optimization (50 trials with Optuna) and saved to `results/xgb_best_params.json`.

| Hyperparameter | Baseline (Default + Class Ratio) | Tuned (Optuna Best) | Description & Impact |
| :--- | :---: | :---: | :--- |
| `scale_pos_weight` | `3.5136236859043124` | `7.2839045017915405` | Balances positive/negative class weights. The ~2x increase heavily penalizes false negatives, surging recall to 92.19%. |
| `learning_rate` | `0.3` *(default)* | `0.047747884330947565` | Step size shrinkage. Lower rate prevents overshooting and leads to smoother, more robust convergence. |
| `n_estimators` | `100` *(default)* | `194` | Number of gradient boosted trees. Increased to compensate for the smaller learning rate. |
| `max_depth` | `6` *(default)* | `4` | Maximum tree depth. Constrained to shallower trees to mitigate overfitting on noisy weather variations. |
| `subsample` | `1.0` *(default)* | `0.9761788582619335` | Subsample ratio of training instances. Introduces stochastic sampling to improve generalization. |

### Key Findings & Trade-offs

- **PR-AUC Improvement (+3.7% relative / +0.0269 absolute):** Tuning improved XGBoost's PR-AUC from `0.7216` to `0.7485`, consolidating its lead as the most accurate classifier across all discrimination thresholds.
- **Massive Recall Surge (+17.8% gain):** Recall jumped from `74.39%` to **`92.19%`**, successfully identifying over 92% of all actual rain events in the next hour. For weather forecasting scenarios, capturing precipitation events and minimizing false negatives is critical.
- **Precision Trade-off:** By optimizing `scale_pos_weight` and tree depth to prioritize capturing minority positive instances, precision adjusted from `58.33%` to `49.39%`, maintaining a solid F1-score of `0.6432`.

---

## Current Status

The base end-to-end pipeline is **complete and functional**:
- API ingestion $\rightarrow$ Data cleaning $\rightarrow$ PostgreSQL storage.
- Time-series feature engineering and target creation.
- Model training (Logistic Regression, Random Forest, XGBoost).
- Bayesian Hyperparameter Optimization with Optuna for XGBoost.
- Automated evaluation metrics generation and visualization exports (`results/`).
- Centralized CLI execution and parameter control via `src.main`.

---

## Roadmap & Next Steps

1. [x] **Model Fine-Tuning:** Hyperparameter optimization using Optuna for XGBoost (Completed).
2. [ ] **API Deployment:** Expose model inference and predictions via a **FastAPI** REST API for real-time scoring.
3. [ ] **Workflow Orchestration:** Schedule and monitor ETL runs and periodic model retraining using **Apache Airflow**.
4. [ ] **CI/CD & Testing:** Implement unit tests (pytest), data quality validation (e.g. Great Expectations), and automated GitHub Actions workflows.

---

## License

This project is intended for educational and portfolio purposes.
