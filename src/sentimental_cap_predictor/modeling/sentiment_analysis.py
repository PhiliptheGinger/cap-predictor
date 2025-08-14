import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path
from colorama import Fore, Style
import typer
import logging
from transformers import DistilBertTokenizer

# Initialize the sentiment analysis pipeline globally for efficiency
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

app = typer.Typer()

def truncate_content(content, max_length=512):
    """Truncate content to fit within the DistilBERT token limit."""
    tokens = tokenizer.encode(content, truncation=True, max_length=max_length)
    truncated_content = tokenizer.decode(tokens, skip_special_tokens=True)
    return truncated_content

def perform_sentiment_analysis(news_df: pd.DataFrame) -> pd.DataFrame:
    """Perform sentiment analysis on the news articles."""
    print(f"{Fore.YELLOW}Starting sentiment analysis on {len(news_df)} articles.{Style.RESET_ALL}")
    
    weighted_sentiments = []
    confidences = []

    if 'content' not in news_df.columns:
        print(f"{Fore.RED}'content' column not found in the DataFrame.{Style.RESET_ALL}")
        return news_df

    news_df['content'] = news_df['content'].fillna("")

    # Handle the date column, using seendate if it's present
    if 'seendate' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['seendate'], errors='coerce')
        print(f"{Fore.CYAN}Using 'seendate' as 'date'.{Style.RESET_ALL}")
    elif 'date' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
        print(f"{Fore.CYAN}Using 'date' column.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}'date' or 'seendate' column not found in the DataFrame.{Style.RESET_ALL}")
        return news_df
    
    news_df['date'] = news_df['date'].dt.normalize()
    print(f"{Fore.CYAN}Date range after conversion: {news_df['date'].min()} to {news_df['date'].max()}{Style.RESET_ALL}")

    for _, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Analyzing sentiment"):
        content = row['content']
        if not content.strip():  # Skip empty content
            logging.warning(f"Skipping article due to empty content: {row.get('title', 'No Title')}")
            weighted_sentiments.append(None)
            confidences.append(None)
            continue

        # Truncate content to fit within the 512-token limit
        truncated_content = truncate_content(content)

        try:
            result = sentiment_pipeline(truncated_content)
            sentiment_label = result[0]['label']
            confidence = result[0]['score']
            
            # Assign weighted sentiment based on sentiment label and confidence
            if sentiment_label == 'POSITIVE':
                weighted_sentiment = confidence
            else:
                weighted_sentiment = -confidence

            # Log the weighted sentiment for debugging
            logging.debug(f"Content: {truncated_content[:30]}..., Sentiment: {sentiment_label}, Confidence: {confidence}, Weighted Sentiment: {weighted_sentiment}")
            
            weighted_sentiments.append(weighted_sentiment)
            confidences.append(confidence)
        except Exception as e:
            logging.error(f"Error processing article: {row.get('title', 'No Title')}. Exception: {e}")
            weighted_sentiments.append(None)
            confidences.append(None)

    news_df['weighted_sentiment'] = weighted_sentiments
    news_df['confidence'] = confidences

    # Replace NaN values in sentiment and confidence with default values
    news_df['weighted_sentiment'].fillna(0, inplace=True)
    news_df['confidence'].fillna(0, inplace=True)

    # Print out rows with NaN sentiment or confidence
    nan_rows = news_df[news_df['weighted_sentiment'].isna() | news_df['confidence'].isna()]
    if not nan_rows.empty:
        print(f"{Fore.RED}Rows with NaN sentiment or confidence:{Style.RESET_ALL}")
        print(nan_rows)

    # Print the first few rows of the final DataFrame
    print(f"{Fore.GREEN}Final DataFrame after sentiment analysis:{Style.RESET_ALL}")
    print(news_df.head())

    return news_df

def aggregate_sentiment_by_date(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment by date, weighted by confidence, to create a single bias factor per day."""
    print(f"{Fore.YELLOW}Aggregating sentiment by date.{Style.RESET_ALL}")
    
    # Calculate weighted average sentiment for each date
    analyzed_df['weighted_confidence'] = analyzed_df['weighted_sentiment'] * analyzed_df['confidence']
    
    # Group by date and calculate the weighted mean sentiment and mean confidence
    aggregated_df = analyzed_df.groupby('date').apply(
        lambda x: pd.Series({
            'bias_factor': x['weighted_confidence'].sum() / x['confidence'].sum() if x['confidence'].sum() != 0 else 0,
            'mean_confidence': x['confidence'].mean()
        })
    ).reset_index()

    # Determine the final sentiment label based on the bias factor
    aggregated_df['final_sentiment'] = aggregated_df['bias_factor'].apply(
        lambda x: 'POSITIVE' if x > 0 else ('NEGATIVE' if x < 0 else 'NEUTRAL')
    )

    print(f"{Fore.GREEN}Final Aggregated DataFrame:{Style.RESET_ALL}")
    print(aggregated_df.head())

    return aggregated_df

@app.command()
def main(news_path: Path, verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output")):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info(f"Loading news data from {news_path}.")
    try:
        news_df = pd.read_feather(news_path)
        logging.info(f"News data loaded successfully with {len(news_df)} articles.")
        
        print(f"{Fore.CYAN}Loaded DataFrame:{Style.RESET_ALL}")
        print(news_df.head())
        
    except Exception as e:
        logging.error(f"Error loading news data from {news_path}: {e}")
        return

    analyzed_df = perform_sentiment_analysis(news_df)

    if analyzed_df is not None and 'weighted_sentiment' in analyzed_df.columns:
        aggregated_df = aggregate_sentiment_by_date(analyzed_df)

        output_path = news_path.with_name(f"{news_path.stem}_analyzed.feather")
        logging.info(f"Saving sentiment analysis results to {output_path}.")
        try:
            aggregated_df.to_feather(output_path)
            logging.info(f"Sentiment analysis results saved to {output_path}.")
            print(f"{Fore.GREEN}Final DataFrame after sentiment analysis:{Style.RESET_ALL}")
            print(aggregated_df.head())  # Print the first few rows of the aggregated DataFrame
        except Exception as e:
            logging.error(f"Error saving sentiment analysis results to {output_path}: {e}")
    else:
        logging.error("Sentiment analysis failed. No sentiment data to save.")

if __name__ == "__main__":
    typer.run(main)
