import numpy as np
import pandas as pd
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)

def add_gaussian_noise(data, std_dev):
    noise = np.random.normal(0, std_dev, data.shape)
    return data + noise

def jitter_data(data, strength):
    jitter = np.random.uniform(-strength, strength, data.shape)
    return data + jitter

def temporal_shift(data, steps):
    return data.shift(periods=steps).fillna(method='bfill')

def first_order_diff(data):
    return data.diff().fillna(method='bfill')

def seasonal_diff(data, period):
    return data.diff(periods=period).fillna(method='bfill')

def apply_augmentations(df, gaussian_std=None, jitter_strength=None, shift_steps=None, first_order_diff_flag=False, seasonal_diff_period=None):
    if gaussian_std is not None:
        logging.info(f"Applying Gaussian noise with std dev {gaussian_std}.")
        df = add_gaussian_noise(df, gaussian_std)
    
    if jitter_strength is not None:
        logging.info(f"Applying jittering with strength {jitter_strength}.")
        df = jitter_data(df, jitter_strength)
    
    if shift_steps is not None:
        logging.info(f"Applying temporal shifts with {shift_steps} steps.")
        df = temporal_shift(df, shift_steps)
    
    if first_order_diff_flag:
        logging.info("Applying first-order differencing.")
        df = first_order_diff(df)
    
    if seasonal_diff_period is not None:
        logging.info(f"Applying seasonal differencing with period {seasonal_diff_period}.")
        df = seasonal_diff(df, seasonal_diff_period)
    
    return df

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply data augmentations to a CSV file.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the augmented CSV file.")
    parser.add_argument("--gaussian_std", type=float, help="Standard deviation for Gaussian noise.")
    parser.add_argument("--jitter_strength", type=float, help="Strength of jittering.")
    parser.add_argument("--shift_steps", type=int, help="Number of steps for temporal shifts.")
    parser.add_argument("--first_order_diff", action="store_true", help="Apply first-order differencing.")
    parser.add_argument("--seasonality_period", type=int, help="Period for seasonal differencing.")
    
    args = parser.parse_args()

    # Load the input CSV file
    df = pd.read_csv(args.input_csv, index_col=0, parse_dates=True)

    # Apply augmentations
    df_augmented = apply_augmentations(
        df,
        gaussian_std=args.gaussian_std,
        jitter_strength=args.jitter_strength,
        shift_steps=args.shift_steps,
        first_order_diff_flag=args.first_order_diff,
        seasonal_diff_period=args.seasonality_period
    )

    # Save the augmented data to the output CSV file
    df_augmented.to_csv(args.output_csv)
    logging.info(f"Augmented data saved to {args.output_csv}")
