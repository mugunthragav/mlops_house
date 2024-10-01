import mlflow
from mlflow.tracking import MlflowClient
from model_training import train_all_models, load_data  # Import necessary functions

mlflow.set_tracking_uri("http://localhost:5000")

def promote_and_register_model():
    # Load the data
    data = load_data()

    # Train models and get the best model
    best_model_name = train_all_models(data)

    # Create an MLflow client
    client = MlflowClient()

    # Register the best model
    with mlflow.start_run():
        # Log the best model
        mlflow.sklearn.log_model(best_model_name,artifact_path="artifacts")

        # Get the latest version of the model from the registry
        model_version = client.get_latest_versions(best_model_name, stages=["None"])[0].version

        # Promote the model to production
        client.transition_model_version_stage(
            name=best_model_name,
            version=model_version,
            stage="Production",
            archive_existing_versions=True
        )

    print(f"Model {best_model_name} promoted to Production!")

if __name__ == "__main__":
    promote_and_register_model()
