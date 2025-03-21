# Imports for FastAPI:
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional

# Imports from existing scripts:
from src.data_loader import create_data 
from src.train import train_model  
from src.predict import predict_single_image 
from src.prod_model_select import update_model_if_better

# here are the links to the files Erwin prepared
# from src.train_mlflow_dagshub import train_model  # import train_model function from train.py
# from src.prod_model_select_mlflow_dagshub import update_model_if_better # import the prod_model_select from the py file

app = FastAPI()

class TrainRequest(BaseModel):
    dataset_path: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Deep Leaf API!"}

@app.post("/train")
async def train_model_endpoint(request: TrainRequest):
    
    # Endpoint that:
    # 1) Merges/loads the data subsets via load_data()
    # 2) Trains the model
    # 3) Checks if the newly trained model is better"

    try:
        create_data() # load data or merge new subsets
        train_model() 
        result = update_model_if_better() # update the model if it’s better (promote to production)

        return {
            "message": "Training completed successfully.",
            "model_management_result": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Training failed: {str(e)}')


@app.post("/predict/")
async def predict_endpoint(file: Optional[UploadFile] = File(None)):

    # Endpoint that:
    # 1) Takes in a file (with the option of giving no input)
    # 2) Checks if the file is provided
    # 3) Checks the file extension
    # 4) Checks the file size
    # 5) Does the prediction

    # Check if file is provided
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    # Example threshold: 5 MB
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
    VALID_EXTENSIONS = {"jpg", "jpeg", "png"}

    # Check file extension
    filename = file.filename
    extension = filename.split(".")[-1].lower()
    if extension not in VALID_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {extension}. Allowed: {VALID_EXTENSIONS}"
        )

    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File size exceeds {MAX_FILE_SIZE} bytes limit."
        )
    # Reset the stream, so predict_single_image can read from the beginning
    await file.seek(0)

    try:
        prediction = predict_single_image(file)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")