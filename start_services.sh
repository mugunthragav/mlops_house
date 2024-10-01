#!/bin/bash

# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow --host 0.0.0.0 --port 5000 &

# Start Streamlit app
streamlit run app.py --server.port 8502 --server.address 0.0.0.0