# Imports for FastAPI:
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Imports from existing scripts:
from src.data_loader import load_data # import the data from data_loader.py
from src.train_mlflow_dagshub import train_model  # import train_model function from train.py
from src.predict import predict_single_image  # import predict_single_image function from predict.py
from src.prod_model_select_mlflow_dagshub import update_model_if_better # import the prod_model_select from the py file

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
        load_data() # load data or merge new subsets
        train_model(dataset_path=request.dataset_path) # train the model, which we can also leave this blank if want no argument to pass
        result = update_model_if_better() # update the model if it’s better (promote to production)

        return {
            "message": "Training completed successfully.",
            "model_management_result": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Training failed: {str(e)}')

# Define /predict Endpoint that takes in an uploaded image, processes it and returns the predicted class label.
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    # Example threshold: 5 MB
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
    VALID_EXTENSIONS = {"jpg", "jpeg", "png"}

    # Check file extension
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
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