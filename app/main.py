from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.train import train_model  # Import train_model function from train.py
from src.predict import predict_single_image  # Import predict_single_image function from predict.py

app = FastAPI()

class TrainRequest(BaseModel):
    dataset_path: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Deep Leaf API!"}

@app.post("/train")
async def train_model_endpoint(request: TrainRequest):
    # train_model(request.dataset_path)
    # REVIEW: the model is not given with the api, the data fetch is implemented in src/data_loader.py and called in train.py
    # REVIEW: I added try/except to get messages if training successfully ended and error if not
    try:
        train_model()
        return{"message": "Model successfully trained."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Training failed: {str(e)}')

# Define /predict endpoint that uses predict_single_image
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    # Call the predict_single_image function
    prediction = predict_single_image(file.filename)  # file.filename is the path or name of the uploaded file
    return {"prediction": prediction}