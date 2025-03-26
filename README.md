# 🌱 Deep Leaf - Plant Disease Classification MLOps Pipeline

## 📌 Overview
**Deep Leaf** is a deep learning-based **image classification pipeline** for detecting plant diseases using **Transfer Learning (VGG16)**. It follows **MLOps best practices**, enabling:
- **Airflow orchestration**
- **MLflow tracking**
- **FastAPI access**
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
├── architecture.excalidraw
├── data
│   ├── raw
│   │   ├── train_subset1.tfrecord
│   │   ├── ...
│   │   ├── train_subset10.tfrecord
│   │   ├── valid_subset1.tfrecord
│   │   ├── ...
│   │   ├── valid_subset10.tfrecord
│   └── training
│       ├── train.tfrecord
│       └── valid.tfrecord
├── data.dvc
├── logs
│   ├── history_20250213_084609.json
│   ├── ...
│   └── history_20250310_201801.json
├── merge_progress.json
├── dockers
│   ├── airflow
│   │   ├── dags
│   │   │   └── dag_train.py
│   │   ├── logs
│   │   └── plugins
│   ├── fastapi
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── mlflow
│       └── Dockerfile
├── models
│   ├── metadata.txt
│   └── production_model.keras
├── production_model.keras.dvc
├── requirements.txt
├── requirements-mac.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── helpers.py
│   ├── model.py
│   ├── predict.py
│   ├── prod_model_select.py
│   ├── prod_model_select_mlflow_dagshub.py (old version for MLflow on dagshub)
│   ├── raw_data_split.py
│   ├── test_config.py
│   ├── train_mlflow_dagshub.py (old version for MLflow on dagshub)
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

**APIs :**

**Description adding at the end** 

**Dagshub :**

**Docker :**

The project uses a containerized architecture with the following breakdown and features:

1. Services Implemented
- Two Postgres database services:
    - `postgres`: for Airflow to store metadata and task information
    - `mlflow-postgres`: for MLflow for tracking experiments
- Redis service for communication between Airflow scheduler and workers:
    - central communication hub for Airflow
    - `redis`: used for Airflow Celery executor as a message broker and result backend 
    -  runs on port 6379
- MLflow server for experiments and tracking:
    - mounted volumes of `mlflow` for: artifacts storage, model directory, logs and temp files
    - depends on `mlflow-postgres` for storage
    - exposes port 5001
- FastAPI-app service to run all executables of the project:
    - used to run web app services (swagger UI)
    - mounted volumes: the entire root directory
    - depends on MLflow service (in order to fetch artifacts and experiment related data)
    - exposes pot 8001 (mapped 800 internally)
- Airflow services, consists of the standard components:
    - `airflow-webserver`: Web interface for Airflow which exposes port 8080
    - `airflow-scheduler`: Manages DAG scheduling
    - `airflow-worker`: Celery workers for distributed task execution
    - `airflow-init`: Initializes Airflow database and creates default user
    - `flower`: Celery monitoring tool which exposes port 5555

2. Comunication and Dependencies
- Airflow Components:
    - Use `redis` as a message broker and result backend
    - Communicate through `celery` for distributed task processing
    - Share common configuration via `x-airflow-common`
- MLflow and FastAPI integration:
    - Share volumes for artifacts, models, and logs
    - FastAPI depends on MLflow service
    - Both use separate PostgreSQL databases
- Database Connections:
    - Airflow uses `postgres` database
    - MLflow uses `mlflow-postgres` database
    - Both databases configured with healthchecks
- Dependencies
    - `postgres` and `redis` must be healthy for Airflow services
    - `mlflow-postgres` must be healthy for MLflow
    - `mlflow` must be running for `fastapi-app`
- Key Volumes and Mounts:
    - Shared project root (.:/app) --> and as applicable
    - Docker socket for DockerOperator in Airflow --> for DAG implementation
- Configuration:
    - Uses environment variables for flexible configuration
    - Mounted `.env` file for testing
    - Basic authentication for Airflow API
- Startup and Initialization
    - Services configured with restart policies
    - Healthchecks ensure proper startup sequence
    - Airflow initialization creates default user

3. Docker-in-Docker for model training within Airflow
- Using `fastapi-app` container for model training
- Enabling running model training as a containerized Airflow task
- Mounting the entire project repository (`.:/app`) into the container
- Using Docker socket volume (/var/run/docker.sock) for container management


Here is an overview:
![Docker Compose Architecture](docker_architecture.png)


**Airflow :**

Using a DAG to run model training for the project with the following Functionality and Features

1. DAG Configuration of `model_training_and_update`
- Scheduled run daily at 3:00 PM and one active run at a time

2. Task Details
- Task ID: train_model
- Container Configuration: 
    - Uses the fastapi-app:latest image
    - Executes python /app/src/train.py command
    - Runs with bridge network mode
    - Auto-removes container after execution

3. Additional feratures
- XCom Push: to captures stdout for logging and tracking

4. Error Handling
- Configured with 1 retry
- 5-minute delay between retries

5. General Remarks:
- Uses DockerOperator for containerized task execution
- Leverages Docker socket for container management
- Enables reproducible and isolated model training workflows


**Streamlit :**
