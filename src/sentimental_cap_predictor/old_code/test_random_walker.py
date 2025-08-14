import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Adjust the import to point to the module where your function is located
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Assuming the main function you're testing is in random_walker.py
from random_walker import main as random_walker_main

# Path to your NFLX.feather file
feather_file_path = r"D:\Programming Projects\CAP\sentimental_cap_predictor\sentimental_cap_predictor\data\raw\NFLX.feather"

@pytest.fixture
def mock_feather_data():
    # Load the feather file data into a DataFrame
    df = pd.read_feather(feather_file_path)

    # Mock pd.read_feather to return this DataFrame when called
    with patch("pandas.read_feather", return_value=df):
        yield df

@pytest.fixture
def mock_input():
    # Mock input to return a fixed ticker symbol (e.g., "NFLX")
    with patch("builtins.input", return_value="NFLX"):
        yield

def test_random_walker_with_mocked_feather(mock_feather_data, mock_input):
    # Set up paths for processed and interim directories
    processed_dir = Path("your_processed_dir_path")
    interim_dir = Path("your_interim_dir_path")

    # Ensure the directories exist (you can mock this if needed)
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Run the main function with the mocked input and mocked feather data
    random_walker_main(interim_dir=interim_dir, processed_dir=processed_dir)

    # Verify that the output file is created
    output_file = processed_dir / "NFLX_final_predictions.csv"  # Adjust for the mocked ticker
    assert output_file.exists(), "Output file was not created"

    # Verify that the output DataFrame is not empty
    df_output = pd.read_csv(output_file)
    assert not df_output.empty, "Output DataFrame is empty"

    # Additional checks can be added to verify the content of df_output
    # For example, checking if certain columns exist or if the values are within expected ranges

# Edge case 1: Missing Values
def test_with_missing_values(mock_input):
    # Create a DataFrame with missing values
    data_with_nans = {
        "Date": pd.date_range(start="2022-01-01", periods=5, freq='D'),
        "Price": [100, np.nan, 105, np.nan, 110],
    }
    df_with_nans = pd.DataFrame(data_with_nans)

    with patch("pandas.read_feather", return_value=df_with_nans):
        test_random_walker_with_mocked_feather(df_with_nans, mock_input)

# Edge case 2: Outliers
def test_with_outliers(mock_input):
    # Create a DataFrame with outliers
    data_with_outliers = {
        "Date": pd.date_range(start="2022-01-01", periods=5, freq='D'),
        "Price": [100, 105, 5000, 105, 110],  # 5000 as an outlier
    }
    df_with_outliers = pd.DataFrame(data_with_outliers)

    with patch("pandas.read_feather", return_value=df_with_outliers):
        test_random_walker_with_mocked_feather(df_with_outliers, mock_input)

# Edge case 3: Empty DataFrame
def test_with_empty_dataframe(mock_input):
    # Create an empty DataFrame
    df_empty = pd.DataFrame(columns=["Date", "Price"])

    with patch("pandas.read_feather", return_value=df_empty):
        test_random_walker_with_mocked_feather(df_empty, mock_input)

# Edge case 4: Duplicated Rows
def test_with_duplicated_rows(mock_input):
    # Create a DataFrame with duplicated rows
    data_with_duplicates = {
        "Date": pd.date_range(start="2022-01-01", periods=3, freq='D'),
        "Price": [100, 105, 110],
    }
    df_with_duplicates = pd.DataFrame(data_with_duplicates)
    df_with_duplicates = pd.concat([df_with_duplicates, df_with_duplicates])  # Duplicate rows

    with patch("pandas.read_feather", return_value=df_with_duplicates):
        test_random_walker_with_mocked_feather(df_with_duplicates, mock_input)

# Edge case 5: Unexpected Data Types
def test_with_unexpected_data_types(mock_input):
    # Create a DataFrame with unexpected data types
    data_with_unexpected_types = {
        "Date": pd.date_range(start="2022-01-01", periods=5, freq='D'),
        "Price": ["100", "105", "110", "invalid_data", "115"],  # Mixing strings and numeric values
    }
    df_with_unexpected_types = pd.DataFrame(data_with_unexpected_types)

    with patch("pandas.read_feather", return_value=df_with_unexpected_types):
        test_random_walker_with_mocked_feather(df_with_unexpected_types, mock_input)

