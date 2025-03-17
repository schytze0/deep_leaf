FROM python:3.8-slim

RUN pip install mlflow psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "postgresql://mlflow:mlflow@mlflow-postgres/mlflow", "--default-artifact-root", "/mlflow/artifacts"]