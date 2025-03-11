# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow[extras] \
    && pip install gunicorn

# Expose MLflow server port
EXPOSE 5000

# Set environment variables (Optional, depending on your DB setup)
# INFO: on mac I have another process on 5000
ENV MLFLOW_TRACKING_URI=http://localhost:5001

# Start the MLflow server with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "mlflow.server:app"]