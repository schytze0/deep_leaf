from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.train import train_model  # Import train_model function from train.py
from app.predict import predict_single_image  # Import predict_single_image function from predict.py

app = FastAPI()

class TrainRequest(BaseModel):
    dataset_path: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Deep Leaf API!"}

@app.post("/train")
async def train_model_endpoint(request: TrainRequest):
    train_model(request.dataset_path)
    return {"message": "Model training started."}

# Define /predict endpoint that uses predict_single_image
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    # Call the predict_single_image function
    prediction = predict_single_image(file.filename)  # file.filename is the path or name of the uploaded file
    return {"prediction": prediction}