from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI()
model = joblib.load("iris_classification_model.joblib")

class IrisFeatures(BaseModel):
    # Standard Python variable names mapped to your special JSON keys using an alias
    sepal_length: float = Field(..., alias="sepal_length (cm)")
    sepal_width: float = Field(..., alias="sepal_width (cm)")
    petal_length: float = Field(..., alias="petal_length (cm)")
    petal_width: float = Field(..., alias="petal_width (cm)")

@app.post("/predict")
def predict_flower(data: IrisFeatures):
    # Access the clean Python variable names here
    input_data = np.array([[
        data.sepal_length, 
        data.sepal_width, 
        data.petal_length, 
        data.petal_width
    ]])
    
    # Generate prediction
    prediction = model.predict(input_data)
    
    # Return the predicted class
    return {"prediction": int(prediction[0])}
