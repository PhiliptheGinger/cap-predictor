import pandas as pd
import pytest
from sentimental_cap_predictor.modeling.bias_predictions import bias_predictions_with_sentiment

@pytest.fixture
def sample_predictions_df():
    data = {
        'date': pd.date_range(start='2024-08-01', periods=10, freq='D'),
        'LSTM_Predictions': [220, 215, 210, 205, 200, 205, 210, 215, 220, 225],
    }
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

@pytest.fixture
def sample_sentiment_df():
    data = {
        'seendate': pd.date_range(start='2024-08-02', periods=3, freq='D'),
        'sentiment': [0.1, -0.2, 0.3],
        'confidence': [0.9, 0.8, 0.7],
    }
    df = pd.DataFrame(data)
    return df

def test_initial_no_nan(sample_predictions_df):
    # Ensure there are no NaN values initially in the LSTM_Predictions column
    assert not sample_predictions_df['LSTM_Predictions'].isna().any(), "Initial NaN values found in LSTM_Predictions column."

def test_bias_predictions_no_nan(sample_predictions_df, sample_sentiment_df):
    # Run the biasing function
    biased_df = bias_predictions_with_sentiment(sample_predictions_df, sample_sentiment_df)
    
    # Check for NaN values in BiasedPrediction column
    assert not biased_df['BiasedPrediction'].isna().any(), "NaN values found in BiasedPrediction column after biasing."

def test_reindexing_no_nan(sample_predictions_df, sample_sentiment_df):
    # Run the biasing function
    biased_df = bias_predictions_with_sentiment(sample_predictions_df, sample_sentiment_df)
    
    # Ensure no NaN values were introduced after reindexing
    reindexed_df = biased_df.reindex(sample_predictions_df.index)
    assert not reindexed_df['BiasedPrediction'].isna().any(), "NaN values found in BiasedPrediction column after reindexing."

def test_bias_predictions_shape(sample_predictions_df, sample_sentiment_df):
    # Run the biasing function
    biased_df = bias_predictions_with_sentiment(sample_predictions_df, sample_sentiment_df)
    
    # Check if the shape of the DataFrame remains consistent
    assert biased_df.shape[0] == sample_predictions_df.shape[0], "The shape of the predictions DataFrame has changed."

def test_bias_predictions_value_change(sample_predictions_df, sample_sentiment_df):
    # Run the biasing function
    biased_df = bias_predictions_with_sentiment(sample_predictions_df, sample_sentiment_df)
    
    # Check if values in BiasedPrediction column have been modified
    original_values = sample_predictions_df['LSTM_Predictions'].values
    biased_values = biased_df['BiasedPrediction'].values
    
    assert not all(original_values == biased_values), "No changes detected in BiasedPrediction column."

def test_no_date_overlap(sample_predictions_df):
    # Create a sentiment DataFrame with no overlapping dates
    sentiment_data = {
        'seendate': pd.date_range(start='2024-07-01', periods=3, freq='D'),
        'sentiment': [0.1, 0.2, 0.3],
        'confidence': [0.9, 0.8, 0.7],
    }
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Run the biasing function
    biased_df = bias_predictions_with_sentiment(sample_predictions_df, sentiment_df)
    
    # Check that the BiasedPrediction column is unchanged if there are no overlapping dates
    assert (biased_df['BiasedPrediction'] == sample_predictions_df['LSTM_Predictions']).all(), "BiasedPrediction column should not change without date overlap."

def test_bias_predictions_with_nan_handling():
    # Create a predictions DataFrame with NaN values
    data = {
        'date': pd.date_range(start='2024-08-01', periods=10, freq='D'),
        'LSTM_Predictions': [220, None, 210, 205, None, 205, 210, None, 220, 225],
    }
    predictions_df = pd.DataFrame(data)
    predictions_df.set_index('date', inplace=True)

    # Create a sentiment DataFrame
    sentiment_data = {
        'seendate': pd.date_range(start='2024-08-02', periods=3, freq='D'),
        'sentiment': [0.1, -0.2, 0.3],
        'confidence': [0.9, 0.8, 0.7],
    }
    sentiment_df = pd.DataFrame(sentiment_data)

    # Run the biasing function
    biased_df = bias_predictions_with_sentiment(predictions_df, sentiment_df)
    
    # Check if NaN values in LSTM_Predictions are handled correctly
    assert not biased_df['BiasedPrediction'].isna().any(), "NaN values found in BiasedPrediction after handling NaN in LSTM_Predictions."

def test_timezone_handling(sample_predictions_df, sample_sentiment_df):
    # Make predictions_df timezone-aware
    sample_predictions_df.index = sample_predictions_df.index.tz_localize('UTC')
    
    # Run the biasing function
    biased_df = bias_predictions_with_sentiment(sample_predictions_df, sample_sentiment_df)
    
    # Ensure there are no NaN values introduced due to timezone handling
    assert not biased_df['BiasedPrediction'].isna().any(), "NaN values found in BiasedPrediction after timezone handling."
