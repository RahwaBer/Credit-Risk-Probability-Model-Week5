from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

# Initialize app
app = FastAPI(title="Credit Risk Prediction API")

# Load model from MLflow Registry
MODEL_NAME = "CreditRiskLogisticModel"
MODEL_STAGE = "Production"  # Or "Staging" if not promoted yet

model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Define the input schema using Pydantic
class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float

@app.get("/")
def read_root():
    return {"message": "Credit Risk Prediction API is live 🚀"}

@app.post("/predict/")
def predict_credit_risk(data: CustomerFeatures):
    # Convert to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    return {
        "is_high_risk": int(prediction),
        "message": "High risk" if prediction == 1 else "Low risk"
    }
