from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Credit Risk Model API"} 

from fastapi import HTTPException
from src.api.pydantic_models import TransactionsRequest, PredictionsResponse, ErrorResponse
import pandas as pd
import logging

from src.predict import predict_risk

@app.post("/predict", response_model=PredictionsResponse, responses={400: {"model": ErrorResponse}})
def predict(transactions_request: TransactionsRequest):
    """
    Predict credit risk for a batch of transactions.
    """
    try:
        # Convert list of TransactionInput to DataFrame
        transactions = [t.dict() for t in transactions_request.transactions]
        if not transactions:
            raise HTTPException(status_code=400, detail="No transactions provided.")
        data_df = pd.DataFrame(transactions)

        # Call prediction pipeline
        results_df = predict_risk(data_df, customer_id=None)
        if results_df is None:
            raise HTTPException(status_code=400, detail="Prediction failed.")

        # Format results for API response
        results = [
            {
                "CustomerId": int(row["CustomerId"]),
                "risk_probability": float(row["risk_probability"]),
                "predicted_high_risk": int(row["predicted_high_risk"])
            }
            for _, row in results_df.iterrows()
        ]
        return {"results": results}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"API prediction error: {e}")
        raise HTTPException(status_code=400, detail="Error during prediction.")
