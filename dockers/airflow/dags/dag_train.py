from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta
import os
from pathlib import Path
from docker.types import Mount

dag_file_dir = os.path.dirname(os.path.abspath(__file__))
host_repo_path = os.path.abspath(os.path.join(dag_file_dir, '../..'))

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 20),  # Updated to today for testing
    'end_date': datetime(2025, 3, 29),   # 10 days from start_date
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'model_training_and_update',
    tags=['airflow', 'deep_leaf'],
    default_args=default_args,
    description='Train model in fastapi-app container',
    schedule_interval='0 15 * * *',  # for daily at 3pm use: '0 15 * * *'
    catchup=False,
    max_active_runs=1,
)

# Task to train the model
train_model = DockerOperator(
    task_id='train_model',
    image='fastapi-app:latest',  # Same as in docker-compose.yaml
    command='python /app/src/train.py',
    docker_url='unix://var/run/docker.sock',
    auto_remove=True,
    network_mode='bridge',
    working_dir='/app',  # ensure Git/DVC operates on repo root
    do_xcom_push=True,       # capture all stdout for XCom in Airflow
    mounts=[
        Mount(source=host_repo_path, target='/app', type='bind') # needs the absolute path!
    ],
    dag=dag,
)
