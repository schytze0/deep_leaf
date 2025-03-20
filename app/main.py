# Imports for FastAPI:
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Imports from existing scripts:
from src.data_loader import load_data 
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
        load_data() # load data or merge new subsets
        train_model() 
        result = update_model_if_better() # update the model if it’s better (promote to production)

        return {
            "message": "Training completed successfully.",
            "model_management_result": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Training failed: {str(e)}')

# Define /predict Endpoint that takes in an uploaded image, processes it and returns the predicted class label.
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
    # Pass the entire UploadFile object, matching predict_single_image in predict.py
        prediction = predict_single_image(file)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")