import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from models.models_config import models  # Import the models dictionary
from mlflow.models import Model
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow:5000")
experiment_name = "House_Price_Prediction_Experiment"
mlflow.set_experiment(experiment_name)

# Initiate  the MLflow client
client = MlflowClient()

def load_data():
    # Load the dataset
    data = pd.read_csv("Housing.csv")
    data = pd.get_dummies(data, drop_first=True)
    return data

def get_train_test_data(data, test_size, random_state):
    # Prepare the train and test data
    X = data.drop("price", axis=1)
    y = data["price"]
    print(f"Training data shape: {X.columns}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_all_models(data):
    best_model_name = None
    best_rmse = float('inf')

    for model_name, model_info in models.items():
        model_class = model_info["class"]
        parameters = model_info["parameters"]
        test_size = model_info["test_size"]
        random_state = model_info["random_state"]

        # Prepare the train/test data
        X_train, X_test, y_train, y_test = get_train_test_data(data, test_size, random_state)

        # Instantiate the model
        model_instance = model_class(**parameters)
        model_instance.fit(X_train, y_train)
        predictions = model_instance.predict(X_test)

        # Calculate RMSE
        rmse = mean_squared_error(y_test, predictions, squared=False)

        # Log model to MLflow
        with mlflow.start_run(nested=True):
            mlflow.sklearn.log_model(model_instance, model_name)
            mlflow.log_params(parameters)
            mlflow.log_metric("rmse", rmse)

            # Register the model
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{model_name}", model_name)

            # Check if this model is the best
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = model_name

        # End the current run
        mlflow.end_run()

    return best_model_name

if __name__ == "__main__":
    data = load_data()
    best_model_name = train_all_models(data)
    print(f"Best model: {best_model_name}")
