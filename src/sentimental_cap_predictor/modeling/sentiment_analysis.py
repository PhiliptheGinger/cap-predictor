# flake8: noqa
import sys
from pathlib import Path

import pandas as pd
import typer
from colorama import Fore, Style
from loguru import logger
from tqdm import tqdm
from transformers import DistilBertTokenizer, pipeline

# Initialize the sentiment analysis pipeline globally for efficiency
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt",
)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

app = typer.Typer()


def truncate_content(content, max_length=512):
    """Truncate content to fit within the DistilBERT token limit."""
    tokens = tokenizer.encode(content, truncation=True, max_length=max_length)
    truncated_content = tokenizer.decode(tokens, skip_special_tokens=True)
    return truncated_content


def perform_sentiment_analysis(news_df: pd.DataFrame) -> pd.DataFrame:
    """Perform sentiment analysis on the news articles."""
    logger.info(
        f"{Fore.YELLOW}Starting sentiment analysis on {len(news_df)} articles."
        f"{Style.RESET_ALL}"
    )

    weighted_sentiments = []
    confidences = []

    if "content" not in news_df.columns:
        logger.error(
            f"{Fore.RED}'content' column not found in the DataFrame.{Style.RESET_ALL}"
        )
        return news_df

    news_df["content"] = news_df["content"].fillna("")

    # Handle the date column, using seendate if it's present
    if "seendate" in news_df.columns:
        news_df["date"] = pd.to_datetime(news_df["seendate"], errors="coerce")
        logger.info(f"{Fore.CYAN}Using 'seendate' as 'date'.{Style.RESET_ALL}")
    elif "date" in news_df.columns:
        news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
        logger.info(f"{Fore.CYAN}Using 'date' column.{Style.RESET_ALL}")
    else:
        logger.error(
            f"{Fore.RED}'date' or 'seendate' column not found in the DataFrame."
            f"{Style.RESET_ALL}"
        )
        return news_df

    news_df["date"] = news_df["date"].dt.normalize()
    logger.info(
        f"{Fore.CYAN}Date range after conversion: {news_df['date'].min()} to "
        f"{news_df['date'].max()}{Style.RESET_ALL}"
    )

    for _, row in tqdm(
        news_df.iterrows(), total=len(news_df), desc="Analyzing sentiment"
    ):
        content = row["content"]
        if not content.strip():  # Skip empty content
            logger.warning(
                f"Skipping article due to empty content: {row.get('title', 'No Title')}"
            )
            weighted_sentiments.append(None)
            confidences.append(None)
            continue

        # Truncate content to fit within the 512-token limit
        truncated_content = truncate_content(content)

        try:
            result = sentiment_pipeline(truncated_content)
            sentiment_label = result[0]["label"]
            confidence = result[0]["score"]

            # Assign weighted sentiment based on sentiment label and confidence
            if sentiment_label == "POSITIVE":
                weighted_sentiment = confidence
            else:
                weighted_sentiment = -confidence

            # Log the weighted sentiment for debugging
            logger.debug(
                "Content: {}..., Sentiment: {}, Confidence: {}, Weighted Sentiment: {}",
                truncated_content[:30],
                sentiment_label,
                confidence,
                weighted_sentiment,
            )

            weighted_sentiments.append(weighted_sentiment)
            confidences.append(confidence)
        except Exception as e:
            logger.error(
                "Error processing article: {}. Exception: {}",
                row.get("title", "No Title"),
                e,
            )
            weighted_sentiments.append(None)
            confidences.append(None)

    news_df["weighted_sentiment"] = weighted_sentiments
    news_df["confidence"] = confidences

    # Replace NaN values in sentiment and confidence with default values
    news_df["weighted_sentiment"].fillna(0, inplace=True)
    news_df["confidence"].fillna(0, inplace=True)

    # Print out rows with NaN sentiment or confidence
    nan_rows = news_df[
        news_df["weighted_sentiment"].isna() | news_df["confidence"].isna()
    ]
    if not nan_rows.empty:
        logger.warning(
            f"{Fore.RED}Rows with NaN sentiment or confidence:{Style.RESET_ALL}"
        )
        logger.warning(nan_rows.to_string())

    # Print the first few rows of the final DataFrame
    logger.info(
        f"{Fore.GREEN}Final DataFrame after sentiment analysis:{Style.RESET_ALL}"
    )
    logger.info(news_df.head())

    return news_df


def aggregate_sentiment_by_date(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment by date, weighted by confidence, to create a single
    bias factor per day."""
    logger.info(f"{Fore.YELLOW}Aggregating sentiment by date.{Style.RESET_ALL}")

    # Calculate weighted average sentiment for each date
    analyzed_df["weighted_confidence"] = (
        analyzed_df["weighted_sentiment"] * analyzed_df["confidence"]
    )

    # Group by date and calculate the weighted mean sentiment
    # and mean confidence
    aggregated_df = (
        analyzed_df.groupby("date").agg(
            bias_factor=(
                "weighted_confidence",
                lambda x: x.sum()
                / analyzed_df.loc[x.index, "confidence"].sum(),
            ),
            mean_confidence=("confidence", "mean"),
        )
    ).reset_index()

    # Determine the final sentiment label based on the bias factor
    aggregated_df["final_sentiment"] = aggregated_df["bias_factor"].apply(
        lambda x: "POSITIVE" if x > 0 else ("NEGATIVE" if x < 0 else "NEUTRAL")
    )

    logger.info(f"{Fore.GREEN}Final Aggregated DataFrame:{Style.RESET_ALL}")
    logger.info(aggregated_df.head())

    return aggregated_df


@app.command()
def main(
    news_path: Path,
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
):
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")

    logger.info(f"Loading news data from {news_path}.")
    try:
        news_df = pd.read_feather(news_path)
        logger.info(
            "News data loaded successfully with {} articles.",
            len(news_df),
        )

        logger.info(f"{Fore.CYAN}Loaded DataFrame:{Style.RESET_ALL}")
        logger.info(news_df.head())

    except Exception as e:
        logger.error(f"Error loading news data from {news_path}: {e}")
        return

    analyzed_df = perform_sentiment_analysis(news_df)

    if analyzed_df is not None and "weighted_sentiment" in analyzed_df.columns:
        aggregated_df = aggregate_sentiment_by_date(analyzed_df)

        output_path = news_path.with_name(f"{news_path.stem}_analyzed.feather")
        logger.info(f"Saving sentiment analysis results to {output_path}.")
        try:
            aggregated_df.to_feather(output_path)
            logger.info(f"Sentiment analysis results saved to {output_path}.")
            logger.info(
                f"{Fore.GREEN}Final DataFrame after sentiment analysis:"
                f"{Style.RESET_ALL}"
            )
            logger.info(aggregated_df.head())
        except Exception as e:
            logger.error(
                "Error saving sentiment analysis results to {}: {}",
                output_path,
                e,
            )
    else:
        logger.error("Sentiment analysis failed. No sentiment data to save.")


if __name__ == "__main__":
    typer.run(main)
