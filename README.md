 MLOps House Price Prediction Project

#Overview
This project implements a complete MLOps workflow for predicting house prices using machine learning in a local setup. It utilizes a CI/CD pipeline to automate model training, promotion, and deployment, along with monitoring using MLflow. The application is built using Streamlit for user interaction.

#Features
Workflow


Data Ingestion: Data is ingested from Housing.csv.
Model Training: The model_training.py script trains multiple machine learning models and logs the best performing model using MLflow.
Model Promotion: The best performing model is promoted to production using the promote_and_register.py script.
Model Serving: The promoted model is deployed and served via a Streamlit web app (app.py).
CI/CD Pipeline: The GitHub Actions workflow ensures continuous integration and continuous delivery (CI/CD), automating testing, code quality checks, Docker builds, and deployment.

Step-by-Step Setup and Execution
1. Clone the Repository
To get started, first clone the repository to your local machine:
  git clone <repository_url>
  cd local_mlops_app
2. Install Dependencies
Ensure that Python 3.9 is installed on your machine. You can install the required dependencies using:
  pip install -r requirements.txt

The project is equiped with a GitHub Actions workflow that automates:

Code Quality Checks: Runs linting and code formatting using black.
Model Training Tests: Executes unit tests on model_training.py using pytest.
Docker Build and Deploy: Builds Docker images and deploys the application automatically.
To trigger the CI/CD pipeline, simply push changes to the main branch of the repository. 
The pipeline will:

Check code quality and formatting.
Run tests for model training.
Build and deploy the Docker containers (MLflow and Streamlit).

The MLflow UI will be accessible at http://localhost:5000 for tracking models and their metrics.
The Streamlit app will be available at http://localhost:8502 for predictions.
