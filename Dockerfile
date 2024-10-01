# Use a base image with Python
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project to the container
COPY . .

# Create a directory for the MLflow database and ensure permissions
RUN mkdir -p /app/mlflow && chmod -R 777 /app/mlflow

# Expose the necessary ports for MLflow and Streamlit
EXPOSE 5000 8502

# Entry point for MLflow and Streamlit
CMD ["bash", "-c", "mlflow ui --host 0.0.0.0 --port 5000 & streamlit run app.py --server.port 8502 --server.address 0.0.0.0"]
