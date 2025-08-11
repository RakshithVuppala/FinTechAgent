import requests
import pandas as pd
import yfinance as yf
import json
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import praw

    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.warning("praw not available - Reddit functionality disabled")

try:
    import feedparser

    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.warning("feedparser not available - RSS functionality disabled")


class FinancialDataCollector:
    """
    Comprehensive financial data collector using free APIs
    """

    def __init__(self):
        """Initialize with API keys from environment variables"""
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.getenv(
            "REDDIT_USER_AGENT", "InvestmentResearchBot/1.0"
        )
        self.fred_api_key = os.getenv("FRED_API_KEY")

        # Initialize Reddit client
        self.reddit = None
        if PRAW_AVAILABLE and all([self.reddit_client_id, self.reddit_client_secret]):
            try:
                self.reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent,
                )
            except Exception as e:
                logger.warning(f"Reddit API initialization failed: {e}")

    def get_yahoo_finance_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Get comprehensive stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period=period)

            result = {
                "ticker": ticker,
                "company_name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "current_price": info.get("currentPrice", 0),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
                "historical_data": hist.tail(30).to_dict("records"),
                "news": stock.news[:5] if hasattr(stock, "news") and stock.news else [],
            }

            logger.info(f"Retrieved Yahoo Finance data for {ticker}")
            return result

        except Exception as e:
            logger.error(f"Error getting Yahoo Finance data for {ticker}: {e}")
            return {}

    def get_sec_filings(
        self, ticker: str, filing_type: str = "10-K", limit: int = 5
    ) -> List[Dict]:
        """Get SEC filings for a company"""
        try:
            headers = {
                "User-Agent": "Investment Research Bot contact@yourcompany.com",
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov",
            }

            # Get company CIK
            cik = self._get_cik(ticker)
            if not cik or cik == "0000000000":
                return []

            cik_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = requests.get(cik_url, headers=headers)
            response.raise_for_status()

            company_data = response.json()
            filings = company_data.get("filings", {}).get("recent", {})

            filing_list = []
            forms = filings.get("form", [])
            filing_dates = filings.get("filingDate", [])
            accession_numbers = filings.get("accessionNumber", [])

            for i, form in enumerate(forms[:50]):
                if filing_type.upper() in form.upper() and len(filing_list) < limit:
                    filing_info = {
                        "form": form,
                        "filing_date": filing_dates[i],
                        "accession_number": accession_numbers[i],
                        "url": f"https://www.sec.gov/Archives/edgar/data/{company_data['cik']}/{accession_numbers[i].replace('-', '')}/{accession_numbers[i]}-index.htm",
                    }
                    filing_list.append(filing_info)

            logger.info(
                f"Retrieved {len(filing_list)} {filing_type} filings for {ticker}"
            )
            return filing_list

        except Exception as e:
            logger.error(f"Error getting SEC filings for {ticker}: {e}")
            return []

    def _get_cik(self, ticker: str) -> str:
        """Get CIK number for a ticker symbol"""
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {"User-Agent": "Investment Research Bot contact@yourcompany.com"}

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            companies = response.json()

            for company_info in companies.values():
                if company_info["ticker"].upper() == ticker.upper():
                    return str(company_info["cik_str"]).zfill(10)

            raise ValueError(f"CIK not found for ticker {ticker}")

        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {e}")
            return "0000000000"

    def get_alpha_vantage_data(self, ticker: str) -> Dict[str, Any]:
        """Get financial data from Alpha Vantage"""
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not provided")
            return {}

        try:
            base_url = "https://www.alphavantage.co/query"

            # Get company overview
            overview_params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": self.alpha_vantage_key,
            }

            overview_response = requests.get(base_url, params=overview_params)
            overview_response.raise_for_status()
            overview_data = overview_response.json()

            # Rate limiting for free tier
            time.sleep(12)  # Free tier allows 5 calls per minute

            # Get income statement
            income_params = {
                "function": "INCOME_STATEMENT",
                "symbol": ticker,
                "apikey": self.alpha_vantage_key,
            }

            income_response = requests.get(base_url, params=income_params)
            income_response.raise_for_status()
            income_data = income_response.json()

            result = {
                "ticker": ticker,
                "overview": overview_data,
                "income_statement": income_data.get("annualReports", [])[:3]
                if "annualReports" in income_data
                else [],
                "key_metrics": {
                    "market_cap": overview_data.get("MarketCapitalization", "N/A"),
                    "pe_ratio": overview_data.get("PERatio", "N/A"),
                    "peg_ratio": overview_data.get("PEGRatio", "N/A"),
                    "dividend_yield": overview_data.get("DividendYield", "N/A"),
                    "profit_margin": overview_data.get("ProfitMargin", "N/A"),
                    "return_on_equity": overview_data.get("ReturnOnEquityTTM", "N/A"),
                },
            }

            logger.info(f"Retrieved Alpha Vantage data for {ticker}")
            return result

        except Exception as e:
            logger.error(f"Error getting Alpha Vantage data for {ticker}: {e}")
            return {}

    def get_fred_economic_data(self, series_ids: List[str] = None) -> Dict[str, Any]:
        """Get economic data from FRED (Federal Reserve Economic Data)"""
        if series_ids is None:
            # Default economic indicators relevant to investment research
            series_ids = [
                "GDP",  # Gross Domestic Product
                "UNRATE",  # Unemployment Rate
                "FEDFUNDS",  # Federal Funds Rate
                "DGS10",  # 10-Year Treasury Rate
            ]

        if not self.fred_api_key:
            logger.warning("FRED API key not provided")
            return {}

        try:
            base_url = "https://api.stlouisfed.org/fred/series/observations"
            economic_data = {}

            for series_id in series_ids:
                params = {
                    "series_id": series_id,
                    "api_key": self.fred_api_key,
                    "file_type": "json",
                    "limit": 10,  # Get last 10 observations
                    "sort_order": "desc",
                }

                response = requests.get(base_url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    observations = data.get("observations", [])
                    if observations:
                        latest_value = observations[0]["value"]
                        latest_date = observations[0]["date"]
                        economic_data[series_id] = {
                            "value": latest_value,
                            "date": latest_date,
                            "series_name": self._get_fred_series_name(series_id),
                        }

                time.sleep(0.1)  # Be respectful to the API

            logger.info(f"Retrieved FRED economic data for {len(economic_data)} series")
            return economic_data

        except Exception as e:
            logger.error(f"Error getting FRED data: {e}")
            return {}

    def _get_fred_series_name(self, series_id: str) -> str:
        """Get human-readable name for FRED series"""
        series_names = {
            "GDP": "Gross Domestic Product",
            "UNRATE": "Unemployment Rate",
            "FEDFUNDS": "Federal Funds Rate",
            "DGS10": "10-Year Treasury Rate",
        }
        return series_names.get(series_id, series_id)

    def get_company_news(self, ticker: str, company_name: str = None) -> List[Dict]:
        """Get recent news about a company using FREE sources"""
        all_articles = []

        # Yahoo Finance News
        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            for article in news[:10]:
                formatted_article = {
                    "title": article.get("title", ""),
                    "description": article.get("summary", ""),
                    "url": article.get("link", ""),
                    "published_at": datetime.fromtimestamp(
                        article.get("providerPublishTime", 0)
                    ).isoformat(),
                    "source": "Yahoo Finance",
                    "provider": article.get("publisher", ""),
                    "type": "yahoo_finance",
                }
                all_articles.append(formatted_article)
        except Exception as e:
            logger.warning(f"Error getting Yahoo Finance news: {e}")

        # RSS Feeds
        if FEEDPARSER_AVAILABLE:
            try:
                rss_feeds = [
                    "https://feeds.bloomberg.com/markets/news.rss",
                    "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
                ]

                for feed_url in rss_feeds:
                    try:
                        feed = feedparser.parse(feed_url)
                        for entry in feed.entries[:5]:
                            content = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
                            if ticker.lower() in content or (
                                company_name and company_name.lower() in content
                            ):
                                article = {
                                    "title": entry.get("title", ""),
                                    "description": entry.get("summary", ""),
                                    "url": entry.get("link", ""),
                                    "published_at": entry.get("published", ""),
                                    "source": "RSS Feed",
                                    "type": "rss_feed",
                                }
                                all_articles.append(article)
                    except Exception as e:
                        continue
                    time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error getting RSS news: {e}")

        return all_articles[:15]

    def get_reddit_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get Reddit sentiment data for a stock (FREE)"""
        if not self.reddit:
            return {}

        try:
            subreddits = ["investing", "stocks", "SecurityAnalysis", "ValueInvesting"]
            all_posts = []

            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    posts = subreddit.search(ticker, time_filter="week", limit=10)

                    for post in posts:
                        post_data = {
                            "title": post.title,
                            "selftext": post.selftext,
                            "score": post.score,
                            "num_comments": post.num_comments,
                            "created_utc": post.created_utc,
                            "subreddit": subreddit_name,
                        }
                        all_posts.append(post_data)
                except Exception as e:
                    continue

            # Simple sentiment analysis
            sentiment_analysis = self._analyze_sentiment(all_posts)

            logger.info(f"Retrieved {len(all_posts)} Reddit posts for {ticker}")
            return sentiment_analysis

        except Exception as e:
            logger.error(f"Error getting Reddit sentiment for {ticker}: {e}")
            return {}

    def _analyze_sentiment(self, posts: List[Dict]) -> Dict[str, Any]:
        """Simple sentiment analysis of Reddit posts"""
        if not posts:
            return {"sentiment": "neutral", "confidence": 0, "post_count": 0}

        positive_keywords = [
            "buy",
            "bullish",
            "up",
            "good",
            "great",
            "strong",
            "growth",
        ]
        negative_keywords = [
            "sell",
            "bearish",
            "down",
            "bad",
            "weak",
            "loss",
            "decline",
        ]

        sentiment_scores = []

        for post in posts:
            text = (post["title"] + " " + post["selftext"]).lower()
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)

            if positive_count > negative_count:
                sentiment_scores.append(1)
            elif negative_count > positive_count:
                sentiment_scores.append(-1)
            else:
                sentiment_scores.append(0)

        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            if avg_sentiment > 0.2:
                overall_sentiment = "positive"
            elif avg_sentiment < -0.2:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
        else:
            overall_sentiment = "neutral"
            avg_sentiment = 0

        return {
            "sentiment": overall_sentiment,
            "sentiment_score": avg_sentiment,
            "post_count": len(posts),
            "sample_posts": posts[:3],
        }

    def collect_all_data(self, ticker: str, company_name: str = None) -> Dict[str, Any]:
        """Collect data from all sources"""
        logger.info(f"Starting data collection for {ticker}")

        result = {
            "ticker": ticker,
            "company_name": company_name,
            "collection_timestamp": datetime.now().isoformat(),
            "data_sources": {},
        }

        # Yahoo Finance
        logger.info("Collecting Yahoo Finance data...")
        result["data_sources"]["yahoo_finance"] = self.get_yahoo_finance_data(ticker)

        # SEC Filings
        logger.info("Collecting SEC filings...")
        result["data_sources"]["sec_filings"] = self.get_sec_filings(ticker)

        # Alpha Vantage (if API key available)
        if self.alpha_vantage_key:
            logger.info("Collecting Alpha Vantage data...")
            result["data_sources"]["alpha_vantage"] = self.get_alpha_vantage_data(
                ticker
            )

        # Economic Data
        if self.fred_api_key:
            logger.info("Collecting economic indicators...")
            result["data_sources"]["economic_data"] = self.get_fred_economic_data()

        # News Data
        logger.info("Collecting news data...")
        result["data_sources"]["news"] = self.get_company_news(ticker, company_name)

        # Reddit Sentiment
        if self.reddit:
            logger.info("Collecting Reddit sentiment...")
            result["data_sources"]["reddit_sentiment"] = self.get_reddit_sentiment(
                ticker
            )

        logger.info(f"Data collection completed for {ticker}")
        return result

    def collect_separated_data(
        self, ticker: str, company_name: str = None
    ) -> Dict[str, Any]:
        """
        Enhanced data collection that separates structured and unstructured data
        """
        logger.info(f"Starting separated data collection for {ticker}")

        # Collect raw data (using your existing method)
        raw_data = self.collect_all_data(ticker, company_name)

        # Separate into structured and unstructured
        structured_data = {
            "ticker": ticker,
            "company_name": company_name,
            "timestamp": datetime.now().isoformat(),
            "financial_metrics": raw_data["data_sources"].get("yahoo_finance", {}),
            "sec_data": raw_data["data_sources"].get("sec_filings", []),
            "economic_indicators": raw_data["data_sources"].get("economic_data", {}),
            "alpha_vantage_metrics": raw_data["data_sources"].get("alpha_vantage", {}),
        }

        unstructured_data = {
            "news_articles": raw_data["data_sources"].get("news", []),
            "reddit_sentiment": raw_data["data_sources"].get("reddit_sentiment", {}),
            "yahoo_news": raw_data["data_sources"]
            .get("yahoo_finance", {})
            .get("news", []),
        }

        return {
            "structured": structured_data,
            "unstructured": unstructured_data,
            "collection_metadata": {
                "total_sources": len(raw_data["data_sources"]),
                "structured_sources": len(
                    [
                        k
                        for k, v in structured_data.items()
                        if v and k not in ["ticker", "company_name", "timestamp"]
                    ]
                ),
                "unstructured_sources": len(
                    [k for k, v in unstructured_data.items() if v]
                ),
            },
        }


# Test the enhanced data collector
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    collector = FinancialDataCollector()

    print("🧪 Testing separated data collection...")
    separated_data = collector.collect_separated_data("AAPL", "Apple Inc.")

    print(f"✅ Structured data keys: {list(separated_data['structured'].keys())}")
    print(f"✅ Unstructured data keys: {list(separated_data['unstructured'].keys())}")
    print(f"📊 Metadata: {separated_data['collection_metadata']}")

    # Show some sample data
    print("\n📈 Sample structured data:")
    financial_metrics = separated_data["structured"]["financial_metrics"]
    if financial_metrics:
        print(f"   Company: {financial_metrics.get('company_name', 'N/A')}")
        print(f"   Current Price: ${financial_metrics.get('current_price', 'N/A')}")
        print(
            f"   Market Cap: ${financial_metrics.get('market_cap', 'N/A'):,}"
            if financial_metrics.get("market_cap")
            else "   Market Cap: N/A"
        )

    print("\n📰 Sample unstructured data:")
    news_articles = separated_data["unstructured"]["news_articles"]
    if news_articles:
        print(f"   News articles found: {len(news_articles)}")
        if len(news_articles) > 0:
            print(f"   Latest: {news_articles[0].get('title', 'No title')[:60]}...")

    print("\n🎉 Separated data collection test completed!")
