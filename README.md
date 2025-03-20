# 🌱 Deep Leaf - Plant Disease Classification MLOps Pipeline

## 📌 Overview
**Deep Leaf** is a deep learning-based **image classification pipeline** for detecting plant diseases using **Transfer Learning (VGG16)**. It follows **MLOps best practices**, enabling:
- **FastAPI access**
- **MLflow tracking**
- **Airflow orchestration**
- **CI/CD with Github**

## Context??
DO WE NEED THIS?

## 📂 Repository Structure

We saved all necessary files for the python runs into the folder `src/`. To use these scripts in the FastAPI (folder `app/`), we create a local package of `src` that is built during installing the requirements (`requirements.txt` or `requirements_mac.txt`). 
The `data/` is once build with the script `raw_data_split.py` and then saved into `data/raw`. Since we simulate new data by adding to the first data set the each of the other data splits (up to 10) we create two new files `train.tfrecord` and `valid.tfrecord` that are saved in `data/training/`.
In `model/`, you can find the current production model (`production_model.keras`) as well as metadata (final validation accuracy score, `metadata.txt`).
In `app/`, you find the creation of the FastAPI.
In `mlflow/`, you can find the creation of the MLflow container. 
In `tests/`, you can find simple test scripts for the unit tests.
There are some helper files:
- `setup.py`: for creation of the package `src` to reference them in `app/`
- `architecture.excalidraw`: visualization of (ongoing) workflow
- `merge_progress.json`: A file to check how far we have been so far with the new data simulation 

```plaintext
.
├── LICENSE
├── README.md
├── app
│   ├── __init__.py
│   └── main.py
├── architecture.excalidraw.png
├── data
│   ├── raw
│   │   ├── train_subset1.tfrecord
│   │   ├── ...
│   │   ├── train_subset10.tfrecord
│   │   ├── valid_subset1.tfrecord
│   │   ├── ...
│   │   └── valid_subset10.tfrecord
│   ├── test
│   │   ├── AppleCedarRust1.JPG
│   │   ├── ...
│   │   ├── AppleCedarRust4.JPG
│   │   ├── AppleScab1.JPG
│   │   ├── AppleScab2.JPG
│   │   ├── AppleScab3.JPG
│   │   ├── CornCommonRust1.JPG
│   │   ├── CornCommonRust2.JPG
│   │   ├── CornCommonRust3.JPG
│   │   ├── PotatoEarlyBlight1.JPG
│   │   ├── ...
│   │   ├── PotatoEarlyBlight5.JPG
│   │   ├── PotatoHealthy1.JPG
│   │   ├── PotatoHealthy2.JPG
│   │   ├── TomatoEarlyBlight1.JPG
│   │   ├── ...
│   │   ├── TomatoEarlyBlight6.JPG
│   │   ├── TomatoHealthy1.JPG
│   │   ├── ...
│   │   ├── TomatoHealthy4.JPG
│   │   ├── TomatoYellowCurlVirus1.JPG
│   │   ├── ...
│   │   └── TomatoYellowCurlVirus6.JPG
│   └── training
│       ├── train.tfrecord
│       └── valid.tfrecord
├── data.dvc
├── docker-compose.yaml
├── dockers
│   ├── airflow
│   │   ├── dags
│   │   │   └── dag_train.py
│   │   ├── logs
│   │   │   ├── ...
│   │   │   ├── dag_processor_manager
│   │   │   │   └── dag_processor_manager.log
│   │   │   └── scheduler
│   │   │       ├── 2025-03-20
│   │   │       │   └── dag_train.py.log
│   │   │       └── latest -> 2025-03-20
│   │   └── plugins
│   ├── fastapi
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── mlflow
│       └── Dockerfile
├── logs
│   ├── ...
│   └── history_20250319_002320.json
├── merge_progress.json
├── mlflow
│   └── artifacts
│       └── ...
├── mlflow.dvc
├── models
│   ├── metadata.txt
│   ├── production_model.keras
│   └── production_model.keras.dvc
├── requirements.txt
├── requirements_mac.txt
├── requirements_wsl2.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── local_dagshub
│   │   ├── data_loader.py
│   │   ├── prod_model_select_mlflow_dagshub.py
│   │   └── train_mlflow_dagshub.py
│   ├── config.py
│   ├── data_loader.py
│   ├── git_dvc_update.py
│   ├── helpers.py
│   ├── model.py
│   ├── predict.py
│   ├── prod_model_select.py
│   ├── raw_data_split.py
│   ├── test_config.py
│   ├── train.py
│   ├── trials.py
│   └── utils.py
├── temp
│   ├── current_accuracy.txt
│   └── current_model.keras
└── tests
    ├── api_server.py
    └── mlflow_server.py
```

## 📈 Data
The original data stems from [Kaggle (New Plant Diseases Dataset)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). Before the project, we downloaded the data set once, and created 10 subsets of training and validation (script `src/raw_data_split.py`), saving the subsets as `.tfrecord` for versioning and then used this 10 subsets as fictional new data income. Therefore, we will add incrementally to the first data set, the other splits to simulate new data incoming. 

## 🧑‍💻 Project Diagram

![Projekt worklfow implementation](architecture.excalidraw.png)

## Application Operation

### FastAPI

*Description*

## MLflow
This project integrates MLflow to track model training, log hyperparameters, and store artifacts for versioning and reproducibility.

1. Tracking and Logging

During training, MLflow logs:
	•	Metrics: Training loss, accuracy, F1-score, validation loss, validation accuracy, and validation F1-score for each epoch.
	•	Hyperparameters: Model type (VGG16), number of epochs, batch size, input shape, and number of classes.
	•	Artifacts: The trained model is saved using mlflow.keras.log_model() for reproducibility.

2. Experiment Setup

Before training starts, MLflow:
	•	Sets the tracking URI from environment variables (MLFLOW_TRACKING_URI).
	•	Defines an experiment name (MLFLOW_EXPERIMENT_NAME).
	•	Initializes default logging for hyperparameters.

3. Logging During Training

A custom callback (MLFlowLogger) logs training metrics at the end of each epoch:
	•	Tracks best validation accuracy and best F1-score across epochs.
	•	Logs final validation metrics after training.

4. Model Storage & Comparison
	•	The trained model is saved in MLflow’s model registry.
	•	The current model is stored locally (temp/current_model.keras) for comparison with previous models.

5. Reproducibility

The training history (accuracy, F1-score, loss) is saved as a JSON file, ensuring results can be analyzed later.

MLflow is set up in a container running the tracking server. We use PostgreSQL database (mlflow-postgres) as backend for tracking experiment metadata. The container stores artifacts in `/app/mflow/artifacts/` which is mounted from the host machine. Access is given via port `5001`.

## Airflow

*Description*

