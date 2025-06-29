# Credit Risk Model

## Project Overview

Bati Bank is partnering with an eCommerce company to enable a buy-now-pay-later service. This project aims to build a Credit Scoring Model using customer transaction data to predict credit risk, assign credit scores, and recommend optimal loan amounts and durations. The solution leverages advanced machine learning, feature engineering, and MLOps best practices.

## Credit Scoring Business Understanding

### Basel II Accord and the Need for Interpretability

The Basel II Capital Accord places strong emphasis on the measurement, management, and disclosure of credit risk. It requires financial institutions to use risk-sensitive approaches for capital adequacy, and to demonstrate that their credit risk models are robust, transparent, and well-documented. This regulatory environment means that our credit scoring model must not only be accurate, but also interpretable and auditable. Stakeholders including regulators, auditors, and internal risk managers must be able to understand how the model arrives at its predictions, justify its use, and explain its decisions to customers if needed. As a result, model documentation, feature engineering transparency, and clear reasoning behind predictions are essential for compliance and trust.

### Proxy Variables for Default and Associated Risks

In many real-world scenarios, especially with new products or limited historical data, a direct "default" label (i.e., a clear indicator of whether a customer failed to repay) may not be available. To train a supervised model, we must therefore create a proxy variable an alternative label that approximates default, such as "overdue by more than 60 days" or "missed three consecutive payments." While this enables model development, it introduces business risks: the proxy may not perfectly capture true default behavior, leading to potential misclassification. If the proxy is too broad or too narrow, the model may overestimate or underestimate risk, resulting in either lost business opportunities (by rejecting good customers) or increased losses (by approving risky customers). Careful design and validation of the proxy variable are thus critical to minimize these risks.

### Trade-offs: Simple vs. Complex Models in Regulated Finance

There is a fundamental trade-off between using simple, interpretable models (such as Logistic Regression with Weight of Evidence encoding) and more complex, high-performance models (like Gradient Boosting Machines):

- **Simple, Interpretable Models:** These models are easier to explain, audit, and document. They allow for clear reasoning about feature importance and decision logic, which is highly valued by regulators and risk committees. However, they may not capture complex, nonlinear relationships in the data, potentially limiting predictive performance.
- **Complex, High-Performance Models:** Advanced algorithms like Gradient Boosting can achieve higher accuracy by modeling intricate patterns and interactions. However, they are often "black boxes," making it difficult to interpret individual predictions or provide clear justifications for decisions. This lack of transparency can be a barrier in regulated environments, where explainability and accountability are paramount.

In practice, the choice depends on the regulatory context, the need for transparency, and the business’s risk appetite. Often, a balance is sought: starting with interpretable models for baseline compliance, and carefully introducing complexity only when it demonstrably improves business outcomes and can be sufficiently explained.

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
