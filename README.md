# Credit Risk Model

## Project Overview

Bati Bank is partnering with an eCommerce company to enable a buy-now-pay-later service. This project aims to build a Credit Scoring Model using customer transaction data to predict credit risk, assign credit scores, and recommend optimal loan amounts and durations. The solution leverages advanced machine learning, feature engineering, and MLOps best practices.

## Credit Scoring Business Understanding

### 1. Basel II Accord's Influence on Model Requirements

The Basel II Accord's emphasis on risk measurement fundamentally shapes our modeling approach in three key ways:
1. **Regulatory Compliance**: The accord requires banks to maintain capital reserves proportional to their risk exposure, necessitating models whose calculations can be transparently audited.

2. **Risk Sensitivity**: Its three-pillar framework (minimum capital requirements, supervisory review, and market discipline) demands models that can precisely differentiate risk levels while maintaining interpretability.

3. **Validation Requirements**: Basel II mandates regular model validation, making documentation essential. Every modeling decision - from variable selection to weight determination - must be justifiable to regulators.

This regulatory environment makes interpretability non-negotiable. While complex models might offer marginally better performance, their "black box" nature could fail to meet supervisory standards for explainability.

## 2. Proxy Variable Necessity and Risks

**Why we need a proxy:**

- The eCommerce partnership is new, so we lack historical default data
- RFM (Recency, Frequency, Monetary) patterns serve as leading indicators of engagement, which correlates with creditworthiness
- Basel II allows alternative data approaches when traditional credit history is unavailable

**Potential business risks:**

1. **Misclassification Risk**: Engaged customers might still default, while disengaged customers might pay reliably
2. **Regulatory Challenge**: Supervisors may question whether the proxy adequately represents true default risk
3. **Fair Lending Concerns**: The proxy might inadvertently discriminate against certain customer segments
4. **Model Drift**: Customer behavior patterns may change over time, requiring frequent proxy recalibration

## 3. Model Complexity Trade-offs

| Consideration        | Simple Model (Logistic Regression + WoE)       | Complex Model (Gradient Boosting)          |
|----------------------|-----------------------------------------------|-------------------------------------------|
| **Interpretability** | High - Clear variable weights                 | Low - Hard to explain predictions         |
| **Performance**      | Moderate - Linear relationships only          | High - Captures complex patterns          |
| **Regulatory Fit**   | Excellent - Industry standard for credit scoring | Challenging - Requires extra documentation |
| **Implementation**   | Easy - Fast scoring, simple monitoring        | Hard - Computational intensive            |
| **Maintenance**      | Low - Stable relationships                    | High - Needs frequent retraining          |

The choice is depends on the regulatory context, the need for transparency, and the business's risk appetite. Often, a balance is sought: starting with interpretable models for baseline compliance, and carefully introducing complexity only when it demonstrably improves business outcomes and can be sufficiently explained.

## Project Structure

```bash

credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                      # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Feature Engineering

Feature engineering is performed in `src/data_processing.py` and the associated Jupyter notebooks. Key steps include:
- **Time-based features:** Extracts hour, day, month, year, and day-of-week from transaction timestamps.
- **Aggregates:** Computes total, average, and standard deviation of transaction amounts and values per customer.
- **Categorical encoding:** Counts unique values and finds the most frequent (mode) for each categorical feature per customer.
- **RFM features:** Calculates Recency, Frequency, and Monetary value for each customer.
- **WOE/IV selection:** Uses Weight of Evidence and Information Value to select the most predictive features.

You can experiment with feature engineering in the provided notebooks, or use the pipeline in `src/data_processing.py` for production-ready transformations.

---

## Model Training

Model training is handled by `src/train.py`. Supported models include:
- Logistic Regression (with or without WoE features)
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

To train a model, run:
```bash
python src/train.py
```
This will split the data, perform hyperparameter tuning, evaluate models, and save the best model artifact to the `model/` directory.

---

## Model Artifacts

- Trained models are saved in the `model/` directory (e.g., `model/best_model.pkl`).
- The pipeline and transformers are saved as `.pkl` files for reuse in prediction and API serving.
- You can load a model in Python with:
  ```python
  import joblib
  model = joblib.load('model/best_model.pkl')
  ```

---

## Manual Testing

### **Manual API Testing**

You can test the FastAPI endpoints manually using `curl` or `httpie`:

- **Root endpoint:**
  ```bash
  curl http://localhost:8000/
  ```
- **Prediction endpoint:**
  ```bash
  http POST http://localhost:8000/predict transactions:='[{"CustomerId": 1, "TransactionId": 101, ...}]'
  ```
  (Replace the JSON with your test data.)

Or use the interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### **Manual Pipeline Testing**

You can test the feature engineering and prediction pipeline in a notebook or script:

```python
from src.data_processing import get_full_preprocessing_pipeline
import pandas as pd
raw_df = pd.read_csv('data/raw/your_data.csv')
pipeline = get_full_preprocessing_pipeline()
processed = pipeline.fit_transform(raw_df)
```

---

## Running the Service with Docker

You can build and run the FastAPI service using Docker or docker-compose.

### **Build and Run with Docker Compose**

```bash
docker-compose up --build
```
- The API will be available at [http://localhost:8000](http://localhost:8000)
- To stop the service:
```bash
docker-compose down
```

### **Build and Run with Docker (manual)**

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

---

## Running Tests and Linting Locally

- **Run all unit tests:**
  ```bash
  pytest
  ```
- **Run code linter (flake8):**
  ```bash
  flake8 src/ --max-line-length=120 --exclude=__init__.py
  ```

---

## CI/CD with GitHub Actions

- Automated CI runs on every push and pull request to the `main` branch.
- The workflow performs:
  1. **Linting** with flake8 (code style must pass)
  2. **Unit testing** with pytest (all tests must pass)
- The build fails if either step fails.

---

## API Usage

Once running, access the API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

- **Root endpoint:** `GET /` returns a welcome message.
- **Prediction endpoint:** `POST /predict` accepts a batch of transactions and returns risk predictions.

See `src/api/main.py` and `src/api/pydantic_models.py` for request/response formats.
