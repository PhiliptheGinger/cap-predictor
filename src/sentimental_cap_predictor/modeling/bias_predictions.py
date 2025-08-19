import pandas as pd
from loguru import logger
import numpy as np

# Add a bias multiplier setting at the top
BIAS_MULTIPLIER = .0001

def bias_predictions_with_sentiment(predictions_df, sentiment_df):
    logger.info("Starting the biasing of predictions with sentiment data.")

    # Normalize sentiment_df columns to lowercase
    sentiment_df.columns = sentiment_df.columns.str.lower()
    logger.debug(f"Normalized sentiment_df columns: {sentiment_df.columns.tolist()}")

    logger.debug(f"Initial columns in predictions_df: {predictions_df.columns.tolist()}")
    logger.debug(f"Initial shape of predictions_df: {predictions_df.shape}")

    # Use 'seendate' as 'date' if necessary
    if 'seendate' in sentiment_df.columns and 'date' not in sentiment_df.columns:
        logger.info("Using 'seendate' as 'date' in sentiment_df.")
        sentiment_df['date'] = pd.to_datetime(sentiment_df['seendate'], errors='coerce')

        # Check the range of dates after conversion to ensure correctness
        logger.debug(f"Converted 'date' range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")

    # Check for the presence of the 'date' column
    if 'date' not in sentiment_df.columns:
        logger.error("'date' column is still missing from sentiment_df after processing.")
        logger.debug(f"Available columns in sentiment_df: {sentiment_df.columns.tolist()}")
        return predictions_df

    # Ensure 'date' is of datetime type and drop NaT values
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce')
    nat_count = sentiment_df['date'].isna().sum()
    if nat_count > 0:
        logger.warning(f"Found {nat_count} NaT entries in 'date' column after datetime conversion.")

    sentiment_df.dropna(subset=['date'], inplace=True)
    sentiment_df.set_index('date', inplace=True)

    logger.debug(f"Sentiment DataFrame index after setting 'date': {sentiment_df.index}")

    # Align the timezone of sentiment_df with predictions_df
    if predictions_df.index.tz is not None:
        # Check if sentiment_df index is already tz-aware
        if sentiment_df.index.tz is not None:
            sentiment_df.index = sentiment_df.index.tz_convert(predictions_df.index.tz)
        else:
            sentiment_df.index = sentiment_df.index.tz_localize('UTC').tz_convert(predictions_df.index.tz)
    else:
        sentiment_df.index = sentiment_df.index.tz_localize(None)  # Remove timezone if not present in predictions_df

    logger.info(f"Sentiment DataFrame shape after date processing: {sentiment_df.shape}")
    logger.debug(f"Sentiment DataFrame index after processing: {sentiment_df.index}")

    # Ensure 'predicted' exists in predictions_df
    if 'predicted' not in predictions_df.columns:
        logger.error("'predicted' column not found in predictions DataFrame.")
        return predictions_df

    # Initialize 'BiasedPrediction' with 'predicted'
    predictions_df['BiasedPrediction'] = predictions_df['predicted'].fillna(0)

    logger.debug(f"Columns after adding 'BiasedPrediction': {predictions_df.columns.tolist()}")
    logger.debug(f"Shape of predictions_df after adding 'BiasedPrediction': {predictions_df.shape}")

    # Create a copy for biasing process
    biased_predictions = predictions_df[['BiasedPrediction']].copy()

    # Ensure 'BiasedPrediction' column can hold float values
    biased_predictions['BiasedPrediction'] = biased_predictions['BiasedPrediction'].astype(float)
    logger.debug(f"Initial biased_predictions shape: {biased_predictions.shape}")

    # Apply biasing based on sentiment data
    for sentiment_date, sentiment_row in sentiment_df.iterrows():
        sentiment_score = sentiment_row.get('sentiment', 0)
        confidence = sentiment_row.get('confidence', 0)
        logger.debug(f"Processing sentiment on {sentiment_date}: sentiment={sentiment_score}, confidence={confidence}")

        start_date = sentiment_date
        end_date = sentiment_date + pd.Timedelta(days=14)
        logger.debug(f"Applying bias from {start_date} to {end_date}")

        # Exponential decay function
        def apply_time_decay(delta_days):
            return np.exp(-delta_days / 14)

        # Iterate over the prediction dates and apply bias
        for pred_date in pd.date_range(start=start_date, end=end_date, freq='D'):
            pred_date = pred_date.normalize()  # Ensure dates are normalized
            if pred_date in biased_predictions.index:
                days_since_sentiment = (pred_date - sentiment_date).days
                time_factor = apply_time_decay(days_since_sentiment)

                adjusted_sentiment = 2 * (sentiment_score - 0.5)
                bias_factor = 1 + adjusted_sentiment * confidence * time_factor * BIAS_MULTIPLIER

                original_prediction = biased_predictions.at[pred_date, 'BiasedPrediction']
                biased_prediction = original_prediction * bias_factor
                
                # Assign and ensure the value is a float
                biased_predictions.at[pred_date, 'BiasedPrediction'] = float(biased_prediction)

                logger.info(f"Date: {pred_date}, Original: {original_prediction}, Bias factor: {bias_factor}, Biased: {biased_prediction}")

    # Ensure all required dates are covered by filling any missing dates
    biased_predictions = biased_predictions.reindex(predictions_df.index, fill_value=predictions_df['predicted'])

    # Assign the final biased predictions back to the original DataFrame
    predictions_df['BiasedPrediction'] = biased_predictions['BiasedPrediction']

    logger.info("Sentiment biasing completed.")
    logger.debug(f"Final biased_predictions shape: {biased_predictions.shape}")
    logger.debug(f"Final columns before returning DataFrame: {predictions_df.columns.tolist()}")
    logger.debug(f"Final shape of predictions_df before returning: {predictions_df.shape}")

    # Ensure the 'BiasedPrediction' column exists before continuing
    if 'BiasedPrediction' not in predictions_df.columns:
        logger.error("BiasedPrediction column is missing after processing.")
        raise KeyError("BiasedPrediction column is missing after processing.")

    return predictions_df
