# Monkey patching the deprecated jax.xla_computation function
import jax
from jax import jit

def patched_xla_computation(f):
    return jit(f).lower().compiler_ir('hlo')

jax.xla_computation = patched_xla_computation

import requests
from transformers import pipeline
import logging
import time
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from newspaper import Article
from tqdm import tqdm
import argparse
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Chroma DB client
chroma_client = chromadb.Client(Settings())
collection: Collection = chroma_client.create_collection(name="news_articles")

# ANSI escape codes for colored output
RED = '\033[91m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
GREEN_BG = '\033[42m'
BLACK_TEXT = '\033[30m'

def get_news(api_url, api_key, stock_symbol):
    try:
        params = {
            'symbol': stock_symbol,
            'from': '2023-01-01',  # Adjust the date range as needed
            'to': '2024-12-31',
            'token': api_key
        }
        logging.info(f"Fetching news for {stock_symbol} from {api_url}")
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            logging.info("Successfully fetched news")
            return response.json()
        else:
            logging.error(f"{RED}Error{RESET}: {BLUE}Failed to fetch news data: {response.status_code}{RESET}")
            logging.error(f"{RED}Error Message{RESET}: {BLUE}{response.content}{RESET}")
            return None
    except Exception as e:
        logging.error(f"Error fetching news: {e}\n{traceback.format_exc()}")
        raise

def article_exists(title, date):
    try:
        logging.info(f"Checking if article exists: {title}")
        results = collection.query(
            query_texts=[title],
            n_results=1,
            where={"date": date}
        )
        exists = len(results['documents']) > 0
        logging.info(f"Article exists: {exists}")
        return exists
    except Exception as e:
        logging.error(f"Error checking article existence: {e}\n{traceback.format_exc()}")
        raise

def sentiment_analysis(text, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
    try:
        # Use PyTorch as the backend
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, framework="pt")
        result = sentiment_pipeline(text)
        logging.info(f"Sentiment analysis result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error performing sentiment analysis: {e}\n{traceback.format_exc()}")
        raise

def fetch_article_content(url):
    try:
        logging.info(f"Fetching article content from URL: {url}")
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logging.error(f"Error fetching article content: {e}\n{traceback.format_exc()}")
        raise

def parse_and_display_news(news_data):
    try:
        if (news_data):
            for i, article in enumerate(news_data):  # Removed the limit on the number of articles
                title = article.get('headline', 'No Title')
                description = article.get('summary', 'No Description')
                date = article.get('datetime', 'No Date')
                link = article.get('url', 'No Link')
                source = article.get('source', 'Unknown Source')

                date_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(date / 1000))  # Convert to seconds

                print(f"\nArticle {i+1}:")
                print(f"{'='*80}")
                print(f"Title      : {title}")
                print(f"Source     : {source}")
                print(f"Date       : {date_str}")
                print(f"Link       : {link}")
                print(f"Description: {description}")
                print(f"{'='*80}\n")

                logging.info(f"Article {i+1}:\nTitle: {title}\nSource: {source}\nDate: {date_str}\nLink: {link}\nDescription: {description}")

                full_content = fetch_article_content(link)
                if full_content:
                    print(f"\nFull Content:\n{full_content[:500]}...")  # Print first 500 characters for brevity
                else:
                    print(f"{RED}Error fetching full content{RESET}")

                sentiment = sentiment_analysis(description)
                if sentiment:
                    sentiment_label = sentiment[0]['label']
                    sentiment_score = sentiment[0]['score']
                    logging.info(f"Sentiment: {sentiment_label}, Confidence: {sentiment_score:.2f}")
                    print(f"\nSentiment Analysis Result:")
                    print(f"{'='*80}")
                    print(f"Sentiment  : {sentiment_label}")
                    print(f"Confidence ≈ {sentiment_score:.2f}")
                    print(f"{'='*80}\n")

                    if not article_exists(title, date_str):
                        collection.add(
                            documents=[{
                                'title': title,
                                'description': description,
                                'date': date_str,
                                'link': link,
                                'source': source,
                                'content': full_content,
                                'sentiment': sentiment_label,
                                'confidence': sentiment_score
                            }]
                        )
                        logging.info(f"Article added to the collection: {title}")
                else:
                    print(f"{RED}Sentiment analysis failed{RESET}")
        else:
            print(f"{RED}No news data available{RESET}")
    except Exception as e:
        logging.error(f"Error parsing and displaying news: {e}\n{traceback.format_exc()}")
        raise

def get_sentiment_score(stock_symbol, test_mode=False):
    try:
        logging.info(f"Test mode: {test_mode}")  # Added logging to verify test mode
        news_api_url = "https://finnhub.io/api/v1/company-news"
        api_key = "cqjer09r01qnjotfl18gcqjer09r01qnjotfl190"

        news_data = get_news(news_api_url, api_key, stock_symbol)

        if news_data:
            sentiment_scores = []
            articles_to_analyze = news_data[:10] if test_mode else news_data
            logging.info(f"Number of articles to analyze: {len(articles_to_analyze)}")  # Log number of articles to analyze
            with tqdm(total=len(articles_to_analyze), desc="Analyzing sentiment", bar_format="{desc} {percentage:3.0f}%|{bar}{r_bar}", colour='green') as pbar:
                for article in articles_to_analyze:
                    start_time = time.time()
                    description = article.get('summary', 'No Description')
                    sentiment = sentiment_analysis(description)
                    if sentiment:
                        sentiment_label = sentiment[0]['label']
                        sentiment_score = sentiment[0]['score']
                        sentiment_scores.append((sentiment_label, sentiment_score))
                        elapsed_time = time.time() - start_time
                        logging.info(f"Time taken for article: {elapsed_time:.2f} seconds")  # Log time taken for each article
                        pbar.set_postfix({"Sentiment": f"{MAGENTA}{sentiment_label}{RESET}", "Confidence": f"{MAGENTA}{sentiment_score:.2f}{RESET}"})
                    pbar.update(1)

            if sentiment_scores:
                positive_scores = [score for label, score in sentiment_scores if label == 'POSITIVE']
                average_positive_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0
                logging.info(f"Average positive sentiment score for {stock_symbol}: {average_positive_score:.2f}")
                return average_positive_score
            else:
                logging.info(f"No positive sentiment scores found for {stock_symbol}")
                return 0
        else:
            logging.info(f"No news data found for {stock_symbol}")
            return 0
    except Exception as e:
        logging.error(f"Error getting sentiment score: {e}\n{traceback.format_exc()}")
        raise

def main(stock_symbol, test_mode=False):
    try:
        score = get_sentiment_score(stock_symbol, test_mode)
        print(f"Sentiment score for {stock_symbol}: {score}")
        return score
    except Exception as e:
        logging.error(f"Error in main function: {e}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentiment Analysis Script')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited data')
    args = parser.parse_args()

    stock_symbol = input("Enter the stock symbol to fetch news for: ").upper()
    main(stock_symbol, test_mode=args.test)
