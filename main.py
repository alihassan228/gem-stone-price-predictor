from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import List,Union
import logging
import pandas as pd
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model Server")

model = joblib.load("gem_price_predictor.pkl")
feature_name = getattr(model,"feature_names_in_",None)
feature_count = getattr(model,"n_features_in_",None)

app = FastAPI()
class predictionRequest (BaseModel):
    rows:List[List[Union[str,float,int,bool]]]

def convert_into_json(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if hasattr(value,"item"):
        return value.item()
    return value

def validate_rows(rows:List[List[Union[str,float,int,bool]]]):
    if len(rows) >500:
        raise HTTPException(400,"Batch size too large. Maximum 500 rows allowed per request")
    
    for index,row in enumerate(rows):
        if feature_count is not None and len(row) !=feature_count:
            raise HTTPException(400,f"Row {index}: Expected {feature_count} features , {len(row)}")
        if None in row:
            raise HTTPException(400,f"Row {index}: Null values are not allowed")
@app.post("/predict")
def predict(request: predictionRequest):
    validate_rows(request.rows)
    input_data = pd.DataFrame(request.rows,columns= feature_name)

    try:
        if hasattr(model,"predict"):
            prediction = model.predict(input_data)
        elif hasattr(model,"transform"):
            prediction = (model.transform(input_data))
        else:
            raise HTTPException(501,"model neither has predict() nor transform()")

        formatted_preds = [f"{val:.2f} USD" for val in prediction]
        response = {"prediction": formatted_preds}
        if hasattr(model,"predict_proba"):
            response["probability"] = convert_into_json(model.predict_proba(input_data))
        return response
    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(500,"Internal error during prediction")

@app.get("/health")
def health():
    return{"status":"ok"}   
 
@app.get("/info")
def info():
    return{
        "model type": type(model).__name__,
        "feature name": list(feature_name) if feature_name is not None
        else None,
        "feature count": feature_count,
        "has predict":hasattr(model,"predict"),
        "has transform":  hasattr(model,"transform"),
        "has predict proba": hasattr(model,"predict_proba")
    }