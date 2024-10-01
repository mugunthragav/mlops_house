import pandas as pd
import pytest
import os

# Define the path to your CSV file
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Housing.csv')  # Adjust the path as necessary


def test_data_format():
    # Check if the CSV file exists
    assert os.path.exists(DATA_FILE_PATH), f"Data file does not exist at {DATA_FILE_PATH}"

    # Attempt to read the CSV file
    try:
        df = pd.read_csv(DATA_FILE_PATH)
    except Exception as e:
        pytest.fail(f"Failed to read the CSV file: {e}")

    # Check if the DataFrame is not empty
    assert not df.empty, "DataFrame is empty. The CSV file might be empty or not formatted correctly."

    # Check if specific columns exist (modify as per your actual columns)
    expected_columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
                        'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
                        'parking', 'prefarea']

    for col in expected_columns:
        assert col in df.columns, f"Expected column '{col}' not found in the data."

    # Optionally check the format of a few rows (e.g., types or values)
    for index, row in df.iterrows():
        assert isinstance(row['price'], (int, float)), "Price should be numeric."
        assert isinstance(row['area'], (int, float)), "Area should be numeric."
        # Add more type checks as needed

