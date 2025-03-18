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
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── main.cpython-311.pyc
│   │   └── main.cpython-38.pyc
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
│   │   ├── ..
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
│   │   ├── logs
│   │   │   ├── dag_processor_manager
│   │   │   │   └── dag_processor_manager.log
│   │   │   └── scheduler
│   │   │       ├── 2025-03-18
│   │   │       └── latest -> 2025-03-18
│   │   └── plugins
│   ├── fastapi
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── mlflow
│       └── Dockerfile
├── logs
│   ├── history_20250213_084609.json
│   └── ...
├── merge_progress.json
├── mlflow
│   └── artifacts
│       └── ...
├── models
│   ├── metadata.txt
│   └── production_model.keras
├── production_model.keras.dvc
├── requirements.txt
├── requirements_mac.txt
├── requirements_wsl2.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── config.py
│   ├── data_loader.py
│   ├── helpers.py
│   ├── local_dagshub
│   │   ├── data_loader.py
│   │   ├── prod_model_select_mlflow_dagshub.py
│   │   └── train_mlflow_dagshub.py
│   ├── model.py
│   ├── predict.py
│   ├── prod_model_select.py
│   ├── prod_model_select_erwin.py
│   ├── raw_data_split.py
│   ├── test_config.py
│   ├── train.py
│   ├── train_erwin.py
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

**FastAPI:**

*Description*

**MLflow:**
We created an independent container to run MLflow. This container saves artifacts and metrics and is linked to local volumes so that data is not lost after shutting down the container. 

**Airflow :**

*Description*

