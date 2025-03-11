# 🌱 Deep Leaf - Plant Disease Classification MLOps Pipeline

## 📌 Overview
**Deep Leaf** is a deep learning-based **image classification pipeline** for detecting plant diseases using **Transfer Learning (VGG16)**. It follows **MLOps best practices**, enabling:
- **Automated dataset handling from Kaggle**
- **Efficient model training & logging**

## 📂 Repository Structure
| File/Folder            | Description |
|------------------------|-------------|
| `src/config.py`           | Stores **global configuration** (paths, credentials, model settings). |
| `src/data_loader.py`      | Handles **dataset downloading & preprocessing**. |
| `src/model.py`            | Defines the **VGG16 transfer learning model**. |
| `src/train.py`            | **Trains the model** in two phases and saves training history. |
| `src/predict.py`          | **Makes predictions** on single images or folders. |
| `src/utils.py`            | Loads & **plots training history** (accuracy & loss). |
| `requirements.txt`    | Lists **dependencies** for setting up the environment. |
| `mac-requirements.txt`    | Lists **dependencies** for setting up the environment with Mac (Silicon, GPU use). |
| `logs/` _(Folder)_    | Stores **training history (`history_*.json`)**. |
| `models/` _(Folder)_  | Stores **trained models (`.keras`)**. (handled with DVC) |
| `data/` _(Folder)_  | Stores **data**. (handled with DVC) |
| `.dvc/` _(Folder)_  | DVC configuration folder |

## 🚀 **Setting Up Deep Leaf for New Developers**
Follow these steps to get started:

### **1️⃣ Fork & Clone the Repositorry**
```sh
git clone https://github.com/schytze0/deep_leaf.git
cd deep_leaf
```

### **2️⃣Fo Create a virtual environment**
Depending on your OS (for example with conda).
```sh
conda create -n my_env python=3.10  
```

### **3️⃣ Install Dependencieses**
```sh
pip install -r requirements.txt
```


### **4️⃣ Set Up Kaggle API Access**
Each team member must store their own Kaggle credentials as GitHub repository secrets.

Step 1: Get Your Kaggle API Key

Go to Kaggle Account Settings.
Click "Create New API Token", which downloads kaggle.json.

Step 2: Add Credentials as GitHub Secrets

Go to GitHub Repo → Settings → Secrets → Actions → New Repository Secret

For each team member, add:

Secret Name	|	Value

KAGGLE_USERNAME_YOURNAME -> "your-kaggle-username"

KAGGLE_KEY_YOURNAME -> "your-kaggle-api-key"


## **🔑 Setting Up the .env File for Automated Environment Setup**

To avoid manually setting environment variables every time, store them in a .env file.

### **1️⃣ Create the .env v File**
Inside the project folder, create a .env file:
```sh
vim .env
```

### **2️⃣ Add the Following Variables to .envnv**
```ini
# User Configuration
GITHUB_ACTOR=your_github_username

# Kaggle API Credentials
KAGGLE_USERNAME_YOURNAME=your_kaggle_username
KAGGLE_KEY_YOURNAME=your_kaggle_api_key
```

## **✅Run the test_config.py file to check the setup**
```sh
python test_config.py
```

There might appear some Tensorflow related warnings (depending on your machine and GPU/CUDA support). The script  should print "Configuration check complete." at the end of the output.

## **🔄 Training the Model**

### Dagshub Credentials
You need to add your dagshub credentials to `.env` before train run:

```sh
DAGSHUB_USERNAME=<your-username>
DAGSHUB_KEY=<your-key>
```

To train the model, run:
```sh
python train.py
```

✔ Downloads dataset from Kaggle.
✔ Trains model in two phases.
✔ Saves best model to models/.
✔ Logs training history in logs/history_*.json.

Instead of the solution above, you can use the train-model that is saved under `models/`. 

## **🔍 Making Predictions**

### **1️⃣ Predict a Single Image**
```sh
python predict.py --image path/to/image.jpg
```

### **2️⃣ Predict a Single Image**
```sh
python predict.py --folder path/to/folder.jpg
```

## **📊 Visualizing Training Performance**
```sh
python utils.py
```

## Running the FastAPI

From within the virtual environment (after you once re-executed requirements to install src), this should work!

```sh
PYTHONPATH=.:$PYTHONPATH uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Curl test

This will test the root endpoint and check if the API is up and running.
```sh
curl -X 'GET' \
  'http://127.0.0.1:8000/' \
  -H 'accept: application/json'
```

This will test the /train endpoint where you pass the dataset path for training. You need to replace your_dataset_path with the actual path to your dataset.
```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/train' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "dataset_path": "../data/raw/train_subset1.tfrecord"
}'
```

This will test the /predict endpoint where you upload an image file for prediction. Replace your_image_file with the actual path to the image file you want to test.
```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -F 'file=@your_image_file'
```

# Work on MLflow as docker (just first notes)

- built folder and Dockerfile
- MLFLOW_TRACKING_URI from Dockerfile written into ENV



```sh
docker build -t mlflow-container . # build docker from mlflow/


# run the docker
docker run -p 5001:5001 --name mlflow-container mlflow-container
```

Airflow DAG

```sh
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mlflow_and_training_pipeline',
    default_args=default_args,
    schedule_interval='@daily',  # Adjust as needed
    start_date=datetime(2025, 3, 11),
    catchup=False
) as dag:

    # Start MLflow container
    start_mlflow_container = DockerOperator(
        task_id='start_mlflow_container',
        image='mlflow-server:latest',  # Name of the MLflow container
        container_name='mlflow-server',
        ports=[5000],
        environment={
            'MLFLOW_TRACKING_URI': 'http://mlflow-server:5000',
            'MLFLOW_EXPERIMENT_NAME': 'your_experiment_name'
        },
        auto_remove=True,
        dag=dag,
    )

    # Train model (in separate container)
    train_model_task = DockerOperator(
        task_id='train_model_task',
        image='your_training_image:latest',  # Image for your training model container
        container_name='training-container',
        environment={
            'MLFLOW_TRACKING_URL': 'http://mlflow-server:5000',  # Connects to MLflow container
            'MLFLOW_EXPERIMENT_NAME': 'your_experiment_name',
            'DATASET_PATH': '/path/to/your/dataset',
        },
        volumes=["/path/to/local/data:/data"],  # Ensure datasets are shared if needed
        auto_remove=True,
        dag=dag,
    )

    start_mlflow_container >> train_model_task
```