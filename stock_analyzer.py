#!/usr/bin/env python3
"""
Stock News Analyzer
Fetches Yahoo Finance news, summarizes key points, and provides AI-powered
buy/sell recommendations based on current market news.

Supports multiple LLM providers via environment variables:
    LLM_PROVIDER: gemini, openai, or anthropic
    LLM_MODEL: (optional) specific model name
    GOOGLE_API_KEY/GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY
"""

import os
import re
import json
from datetime import datetime
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from dotenv import load_dotenv

from llm_provider import generate, get_provider_info, LLMError, reset_provider

# Load environment variables
load_dotenv()


class YahooFinanceNewsFetcher:
    """Fetches news from Yahoo Finance."""

    RSS_FEEDS = {
        "top_news": "https://finance.yahoo.com/news/rssindex",
        "stock_market": "https://finance.yahoo.com/rss/topstories",
    }

    TRENDING_URL = "https://finance.yahoo.com/trending-tickers"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })

    def fetch_rss_news(self, limit: int = 20) -> list[dict]:
        """Fetch news from Yahoo Finance RSS feeds."""
        all_news = []

        for feed_name, feed_url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:limit]:
                    all_news.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": feed_name
                    })
            except Exception as e:
                print(f"Error fetching {feed_name}: {e}")

        return all_news

    def fetch_trending_tickers(self) -> list[dict]:
        """Fetch trending stock tickers from Yahoo Finance."""
        try:
            response = self.session.get(self.TRENDING_URL, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            tickers = []
            ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')

            tables = soup.find_all("table")
            for table in tables:
                rows = table.find_all("tr")
                for row in rows[1:10]:
                    cells = row.find_all("td")
                    if cells:
                        symbol = cells[0].get_text(strip=True)
                        if ticker_pattern.match(symbol) and len(symbol) <= 5:
                            tickers.append({"symbol": symbol})

            if not tickers:
                tickers = [
                    {"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": "GOOGL"},
                    {"symbol": "AMZN"}, {"symbol": "NVDA"}, {"symbol": "META"},
                    {"symbol": "TSLA"}, {"symbol": "JPM"}, {"symbol": "V"},
                    {"symbol": "JNJ"}
                ]

            return tickers[:10]
        except Exception as e:
            print(f"Error fetching trending tickers: {e}")
            return []

    def get_stock_info(self, symbol: str) -> dict:
        """Get current stock information using yfinance."""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="5d")

            current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            prev_close = info.get("previousClose", current_price)

            change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0

            return {
                "symbol": symbol,
                "name": info.get("shortName", symbol),
                "current_price": round(current_price, 2) if current_price else 0,
                "previous_close": round(prev_close, 2) if prev_close else 0,
                "change_percent": round(change_pct, 2),
                "market_cap": info.get("marketCap", 0),
                "volume": info.get("volume", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "sector": info.get("sector", "Unknown"),
            }
        except Exception as e:
            print(f"Error fetching stock info for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    def fetch_stock_news(self, symbol: str, limit: int = 5) -> list[dict]:
        """Fetch news for a specific stock."""
        try:
            stock = yf.Ticker(symbol)
            news = stock.news[:limit] if stock.news else []

            return [{
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0)
                ).strftime("%Y-%m-%d %H:%M") if item.get("providerPublishTime") else "",
                "symbol": symbol
            } for item in news]
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []


class StockAnalyzer:
    """Uses configurable LLM to analyze stock news and provide recommendations."""

    def __init__(self):
        """Initialize analyzer. LLM is configured via environment variables."""
        self._provider_info = None

    def get_provider_info(self) -> dict:
        """Get current LLM provider information."""
        if self._provider_info is None:
            self._provider_info = get_provider_info()
        return self._provider_info

    def analyze(self, news_data: list[dict], stock_data: list[dict]) -> dict:
        """Analyze news and stock data to provide buy/sell recommendations."""
        try:
            news_summary = self._format_news(news_data)
            stock_summary = self._format_stocks(stock_data)

            prompt = f"""You are an expert financial analyst. Analyze the following market news and stock data to provide actionable investment recommendations.

## Recent Market News:
{news_summary}

## Current Stock Data:
{stock_summary}

Based on this information, provide:

1. **Market Summary**: A brief overview of current market conditions and sentiment (2-3 sentences)

2. **Key Takeaways**: The 3-5 most important points from the news that could affect stock prices

3. **BUY Recommendations**: List 2-3 stocks to consider buying with reasoning based on the news
   - Include the ticker symbol, brief reason, and confidence level (High/Medium/Low)

4. **SELL Recommendations**: List 2-3 stocks to consider selling or avoiding with reasoning
   - Include the ticker symbol, brief reason, and confidence level (High/Medium/Low)

5. **Watch List**: 2-3 stocks to monitor closely in the coming days

6. **Risk Disclaimer**: Brief reminder about investment risks

Format your response in clear markdown sections. Be specific and reference actual news items in your analysis."""

            response = generate(prompt)
            provider_info = self.get_provider_info()

            return {
                "success": True,
                "analysis": response,
                "provider": provider_info.get("provider"),
                "model": provider_info.get("model"),
                "timestamp": datetime.now().isoformat()
            }

        except LLMError as e:
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _format_news(self, news_data: list[dict]) -> str:
        """Format news data for the prompt."""
        formatted = []
        for i, news in enumerate(news_data[:15], 1):
            title = news.get("title", "No title")
            summary = news.get("summary", "")[:200]
            published = news.get("published", "")
            formatted.append(f"{i}. **{title}**\n   {summary}\n   Published: {published}")
        return "\n\n".join(formatted) if formatted else "No recent news available."

    def _format_stocks(self, stock_data: list[dict]) -> str:
        """Format stock data for the prompt."""
        formatted = []
        for stock in stock_data:
            if "error" not in stock:
                formatted.append(
                    f"- **{stock['symbol']}** ({stock.get('name', 'N/A')}): "
                    f"${stock.get('current_price', 0)} "
                    f"({stock.get('change_percent', 0):+.2f}%) | "
                    f"P/E: {stock.get('pe_ratio', 'N/A')} | "
                    f"Sector: {stock.get('sector', 'N/A')}"
                )
        return "\n".join(formatted) if formatted else "No stock data available."


# Backward compatibility alias
GeminiStockAnalyzer = StockAnalyzer


class StockNewsApp:
    """Main application class."""

    def __init__(self):
        self.fetcher = YahooFinanceNewsFetcher()
        self.analyzer = StockAnalyzer()

    def run(self, custom_tickers: Optional[list[str]] = None, progress_callback=None) -> dict:
        """Run the full analysis pipeline."""

        def update_progress(message):
            if progress_callback:
                progress_callback(message)
            print(message)

        update_progress("Fetching trending tickers...")
        if custom_tickers:
            tickers = [{"symbol": t.upper()} for t in custom_tickers]
        else:
            tickers = self.fetcher.fetch_trending_tickers()

        ticker_symbols = [t["symbol"] for t in tickers[:10]]

        update_progress("Fetching market news...")
        general_news = self.fetcher.fetch_rss_news(limit=15)

        update_progress("Fetching stock data...")
        stock_data = []
        stock_news = []

        for symbol in ticker_symbols:
            info = self.fetcher.get_stock_info(symbol)
            stock_data.append(info)

            news = self.fetcher.fetch_stock_news(symbol, limit=3)
            stock_news.extend(news)

        all_news = general_news + stock_news

        provider_info = self.analyzer.get_provider_info()
        update_progress(f"Running AI analysis with {provider_info.get('provider', 'LLM')}...")

        analysis = self.analyzer.analyze(all_news, stock_data)

        return {
            "tickers": ticker_symbols,
            "news": general_news,
            "news_count": len(all_news),
            "stock_data": stock_data,
            "analysis": analysis,
            "provider_info": provider_info,
            "timestamp": datetime.now().isoformat()
        }


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stock News Analyzer - Multi-provider LLM support",
        epilog="""
Environment Variables:
  LLM_PROVIDER          gemini, openai, or anthropic (required)
  LLM_MODEL             Specific model name (optional)
  GOOGLE_API_KEY        API key for Gemini
  GEMINI_API_KEY        Alternative API key for Gemini
  OPENAI_API_KEY        API key for OpenAI
  ANTHROPIC_API_KEY     API key for Anthropic

Example:
  LLM_PROVIDER=openai LLM_MODEL=gpt-4o python stock_analyzer.py -t AAPL MSFT
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tickers", "-t",
        nargs="+",
        help="Custom list of stock tickers to analyze (e.g., AAPL MSFT GOOGL)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("STOCK NEWS ANALYZER")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check provider configuration
    provider_info = get_provider_info()
    if provider_info.get("status") == "error":
        print(f"\nConfiguration Error: {provider_info.get('error')}")
        print("\nSet LLM_PROVIDER and the appropriate API key environment variable.")
        print("Supported providers: gemini, openai, anthropic")
        return

    print(f"Provider: {provider_info.get('provider')} | Model: {provider_info.get('model')}")
    print("=" * 60)

    app = StockNewsApp()
    results = app.run(custom_tickers=args.tickers)

    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    if results["analysis"].get("success"):
        print(results["analysis"]["analysis"])
    else:
        print(f"\nError: {results['analysis'].get('error')}")

    output_file = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
