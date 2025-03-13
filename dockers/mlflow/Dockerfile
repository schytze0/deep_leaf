# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow[extras] 

# Expose MLflow server port
EXPOSE 5001

# Set environment variables (Optional, depending on your DB setup)
# INFO: on mac I have another process on 5000
ENV MLFLOW_TRACKING_URI=http://mlflow:5001

# Start the MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/app/mlruns"]