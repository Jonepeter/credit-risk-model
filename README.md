# Credit Risk Model

## Project Overview

Bati Bank is partnering with an eCommerce company to enable a buy-now-pay-later service. This project aims to build a Credit Scoring Model using customer transaction data to predict credit risk, assign credit scores, and recommend optimal loan amounts and durations. The solution leverages advanced machine learning, feature engineering, and MLOps best practices.

## Credit Scoring Business Understanding

### 1. Basel II Accord's Influence on Model Requirements

<<<<<<< HEAD
The Basel II Capital Accord places strong emphasis on the measurement, management, and disclosure of credit risk. It requires financial institutions to use risk-sensitive approaches for capital adequacy, and to demonstrate that their credit risk models are robust, transparent, and well-documented. This regulatory environment means that our credit scoring model must not only be accurate, but also interpretable and auditable. Stakeholders including regulators, auditors, and internal risk managers must be able to understand how the model arrives at its predictions, justify its use, and explain its decisions to customers if needed. As a result, model documentation, feature engineering transparency, and clear reasoning behind predictions are essential for compliance and trust.
=======
The Basel II Accord's emphasis on risk measurement fundamentally shapes our modeling approach in three key ways:
>>>>>>> task-1

1. **Regulatory Compliance**: The accord requires banks to maintain capital reserves proportional to their risk exposure, necessitating models whose calculations can be transparently audited.

<<<<<<< HEAD
In many real-world scenarios, especially with new products or limited historical data, a direct "default" label (i.e., a clear indicator of whether a customer failed to repay) may not be available. To train a supervised model, we must therefore create a proxy variable an alternative label that approximates default, such as "overdue by more than 60 days" or "missed three consecutive payments." While this enables model development, it introduces business risks: the proxy may not perfectly capture true default behavior, leading to potential misclassification. If the proxy is too broad or too narrow, the model may overestimate or underestimate risk, resulting in either lost business opportunities (by rejecting good customers) or increased losses (by approving risky customers). Careful design and validation of the proxy variable are thus critical to minimize these risks.
=======
2. **Risk Sensitivity**: Its three-pillar framework (minimum capital requirements, supervisory review, and market discipline) demands models that can precisely differentiate risk levels while maintaining interpretability.
>>>>>>> task-1

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

The choice is depends on the regulatory context, the need for transparency, and the business’s risk appetite. Often, a balance is sought: starting with interpretable models for baseline compliance, and carefully introducing complexity only when it demonstrably improves business outcomes and can be sufficiently explained.

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
