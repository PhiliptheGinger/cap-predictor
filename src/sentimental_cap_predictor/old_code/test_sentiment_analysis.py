import pytest
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
from sentimental_cap_predictor.modeling.bias_predictions import bias_predictions_with_sentiment, generate_predictions

@pytest.fixture
def mock_data():
    # Create a mock predictions DataFrame
    date_rng = pd.date_range(start='2024-08-01', end='2024-08-31', freq='D')
    predictions_df = pd.DataFrame(date_rng, columns=['date'])
    predictions_df.set_index('date', inplace=True)
    predictions_df['SARIMA'] = np.random.randn(len(date_rng))
    predictions_df['LSTM_Predictions'] = np.random.randn(len(date_rng))
    
    # Create a mock sentiment DataFrame
    sentiment_data = {
        'date': [datetime(2024, 8, 10), datetime(2024, 8, 20)],
        'sentiment': [0.5, -0.7],
        'confidence': [0.8, 0.9],
        'providerpublishtime': [datetime(2024, 8, 10).timestamp(), datetime(2024, 8, 20).timestamp()]
    }
    sentiment_df = pd.DataFrame(sentiment_data)
    
    return predictions_df, sentiment_df

def test_bias_predictions_with_sentiment(mock_data):
    predictions_df, sentiment_df = mock_data
    
    biased_predictions_df = bias_predictions_with_sentiment(predictions_df.copy(), sentiment_df)
    
    # Ensure that the BiasedPrediction column is created
    assert 'BiasedPrediction' in biased_predictions_df.columns, "BiasedPrediction column is missing"

    # Ensure that the biasing has applied some change
    assert not biased_predictions_df['BiasedPrediction'].equals(predictions_df['LSTM_Predictions']), "Biasing did not alter the predictions"
    
    # Check that all necessary columns exist after processing
    required_columns = ['SARIMA', 'LSTM_Predictions', 'BiasedPrediction']
    assert all(col in biased_predictions_df.columns for col in required_columns), "One or more required columns are missing in the biased predictions DataFrame"

def test_generate_predictions(mock_data):
    predictions_df, sentiment_df = mock_data
    
    final_df = generate_predictions(predictions_df.copy(), sentiment_df)
    
    # Ensure that the final DataFrame contains the required columns
    required_columns = ['SARIMA', 'LSTM_Predictions', 'BiasedPrediction']
    assert all(col in final_df.columns for col in required_columns), "One or more required columns are missing in the final DataFrame"

    # Ensure no NaN values exist in the final BiasedPrediction column
    assert final_df['BiasedPrediction'].notna().all(), "BiasedPrediction column contains NaN values"

if __name__ == "__main__":
    pytest.main()
