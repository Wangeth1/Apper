#!/usr/bin/env python3
"""
Auto Trading Analyzer - Streamlit Web Interface
Automatically analyzes news, generates trade recommendations, and paper trades.

Uses a self-contained local NLP engine (no external AI APIs required).
Auto-trades during market hours on every news refresh.
"""

import os
import streamlit as st
from datetime import datetime, timedelta
import feedparser
import requests
from bs4 import BeautifulSoup
import json
import pytz
import yfinance as yf
import re
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from local_analyzer import LocalTradingEngine

# File path for persistent storage
PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio_data.json")

# Page configuration
st.set_page_config(
    page_title="Auto Trading Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .profit { color: #00c853; font-weight: bold; }
    .loss { color: #ff1744; font-weight: bold; }
    .neutral { color: #666; }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
    }
    .loading-indicator {
        position: fixed;
        top: 60px;
        right: 20px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        border-radius: 8px;
        padding: 6px 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 8px;
        color: #333;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .spinner-icon {
        width: 16px;
        height: 16px;
        border: 2px solid #ddd;
        border-top: 2px solid #1f77b4;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Constants
EASTERN_TZ = pytz.timezone('US/Eastern')
INITIAL_CAPITAL = 100000.00  # $100,000 starting capital

NEWS_FEEDS = {
    "Top News": "https://finance.yahoo.com/news/rssindex",
    "Market News": "https://finance.yahoo.com/rss/topstories",
    "Stock Market": "https://finance.yahoo.com/rss/stock-market-news",
}

# Popular stocks from NYSE and NASDAQ for trading
TRADEABLE_STOCKS = [
    # NASDAQ
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC",
    "PYPL", "ADBE", "CSCO", "CMCSA", "PEP", "COST", "TMUS", "AVGO", "TXN", "QCOM",
    # NYSE
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC",
    "XOM", "CVX", "KO", "PFE", "MRK", "ABT", "VZ", "T", "NKE", "MCD"
]


def is_market_open():
    """Check if US stock market is currently open (NYSE/NASDAQ hours)."""
    now = datetime.now(EASTERN_TZ)

    # Market is closed on weekends
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False, "Market closed (Weekend)"

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    if now < market_open:
        return False, f"Market opens at 9:30 AM ET (in {market_open - now})"
    elif now > market_close:
        return False, "Market closed for today (after 4:00 PM ET)"
    else:
        return True, "Market is OPEN"


def get_stock_price(symbol):
    """Get current stock price from Yahoo Finance."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if price:
            return {
                'symbol': symbol,
                'price': round(float(price), 2),
                'name': info.get('shortName', symbol),
                'exchange': info.get('exchange', 'Unknown'),
                'success': True
            }
        return {'symbol': symbol, 'success': False, 'error': 'Price not available'}
    except Exception as e:
        return {'symbol': symbol, 'success': False, 'error': str(e)}


def get_multiple_prices(symbols):
    """Get prices for multiple symbols efficiently."""
    prices = {}
    for symbol in symbols:
        result = get_stock_price(symbol)
        if result['success']:
            prices[symbol] = result['price']
    return prices


@st.cache_data(ttl=30)
def fetch_news(categories=None):
    """Fetch news from RSS feeds."""
    all_news = []
    feeds_to_fetch = {k: v for k, v in NEWS_FEEDS.items() if categories is None or k in categories}

    for category, feed_url in feeds_to_fetch.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:8]:
                all_news.append({
                    "title": entry.get("title", ""),
                    "summary": BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:500],
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "category": category,
                })
        except Exception as e:
            print(f"Error fetching {category}: {e}")

    return {"news": all_news, "news_count": len(all_news), "timestamp": datetime.now().isoformat()}


def generate_trade_recommendations(news_data):
    """Generate trade recommendations using local NLP engine (no external API)."""
    try:
        engine = LocalTradingEngine(
            tradeable_stocks=TRADEABLE_STOCKS,
            min_price=8.0,
            decay_half_life=6.0,
        )

        # Convert news_data dicts into the format the engine expects
        stories = []
        for item in news_data[:15]:
            stories.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "published": item.get("published", ""),
            })

        results = engine.analyze_stories(stories)
        return results

    except Exception as e:
        return {"success": False, "error": str(e)}


def save_portfolio():
    """Save portfolio data to file for persistence."""
    try:
        data = {
            'portfolio': st.session_state.portfolio,
            'recommendation_history': st.session_state.recommendation_history,
            'current_recommendations': st.session_state.get('current_recommendations'),
            'last_news_hash': st.session_state.get('last_news_hash'),
            'saved_at': datetime.now().isoformat()
        }
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving portfolio: {e}")


def load_portfolio():
    """Load portfolio data from file."""
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r') as f:
                data = json.load(f)
                return data
    except Exception as e:
        print(f"Error loading portfolio: {e}")
    return None


def initialize_portfolio():
    """Initialize the paper trading portfolio in session state."""
    if 'portfolio' not in st.session_state:
        # Try to load from file first
        saved_data = load_portfolio()
        if saved_data and 'portfolio' in saved_data:
            st.session_state.portfolio = saved_data['portfolio']
            st.session_state.recommendation_history = saved_data.get('recommendation_history', [])
            if saved_data.get('current_recommendations'):
                st.session_state.current_recommendations = saved_data['current_recommendations']
            if saved_data.get('last_news_hash'):
                st.session_state.last_news_hash = saved_data['last_news_hash']
        else:
            st.session_state.portfolio = {
                'cash': INITIAL_CAPITAL,
                'positions': {},  # {symbol: {'shares': X, 'avg_cost': Y}}
                'trade_history': [],
                'initial_capital': INITIAL_CAPITAL,
                'created_at': datetime.now().isoformat()
            }
    if 'recommendation_history' not in st.session_state:
        st.session_state.recommendation_history = []


def execute_paper_trade(symbol, action, amount_dollars, current_price, reason, confidence):
    """Execute a paper trade if market is open."""
    market_open, market_status = is_market_open()

    if not market_open:
        return {
            'success': False,
            'error': f"Cannot trade: {market_status}",
            'market_status': market_status
        }

    portfolio = st.session_state.portfolio

    if action == "BUY":
        if amount_dollars > portfolio['cash']:
            amount_dollars = portfolio['cash']  # Use available cash

        if amount_dollars < 1:
            return {'success': False, 'error': 'Insufficient funds'}

        shares = amount_dollars / current_price

        # Update position
        if symbol in portfolio['positions']:
            pos = portfolio['positions'][symbol]
            total_shares = pos['shares'] + shares
            total_cost = (pos['shares'] * pos['avg_cost']) + (shares * current_price)
            pos['shares'] = total_shares
            pos['avg_cost'] = total_cost / total_shares
        else:
            portfolio['positions'][symbol] = {
                'shares': shares,
                'avg_cost': current_price
            }

        portfolio['cash'] -= amount_dollars

    elif action == "SELL":
        if symbol not in portfolio['positions'] or portfolio['positions'][symbol]['shares'] <= 0:
            return {'success': False, 'error': f'No position in {symbol} to sell'}

        pos = portfolio['positions'][symbol]
        shares_to_sell = min(amount_dollars / current_price, pos['shares'])
        sale_value = shares_to_sell * current_price

        pos['shares'] -= shares_to_sell
        portfolio['cash'] += sale_value

        if pos['shares'] < 0.0001:  # Clean up tiny positions
            del portfolio['positions'][symbol]

        shares = shares_to_sell

    # Record trade
    trade_record = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'action': action,
        'shares': shares,
        'price': current_price,
        'amount': amount_dollars,
        'reason': reason,
        'confidence': confidence,
        'portfolio_value_at_trade': calculate_portfolio_value()
    }

    portfolio['trade_history'].append(trade_record)

    # Save portfolio after each trade
    save_portfolio()

    return {'success': True, 'trade': trade_record}


def calculate_portfolio_value():
    """Calculate current total portfolio value."""
    portfolio = st.session_state.portfolio
    total = portfolio['cash']

    if portfolio['positions']:
        symbols = list(portfolio['positions'].keys())
        prices = get_multiple_prices(symbols)

        for symbol, pos in portfolio['positions'].items():
            if symbol in prices:
                total += pos['shares'] * prices[symbol]

    return total


def calculate_position_pnl(symbol, current_price):
    """Calculate P&L for a specific position."""
    portfolio = st.session_state.portfolio
    if symbol not in portfolio['positions']:
        return 0, 0

    pos = portfolio['positions'][symbol]
    current_value = pos['shares'] * current_price
    cost_basis = pos['shares'] * pos['avg_cost']
    pnl = current_value - cost_basis
    pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else 0

    return pnl, pnl_percent


def get_news_hash(news_data):
    """Generate a hash of news titles to detect changes."""
    import hashlib
    titles = sorted([item.get('title', '') for item in news_data])
    combined = '|'.join(titles)
    return hashlib.md5(combined.encode()).hexdigest()


def auto_execute_trades(recommendations):
    """Automatically execute trades from NLP recommendations."""
    executed_trades = []
    portfolio = st.session_state.portfolio

    for rec in recommendations.get('recommendations', []):
        symbol = rec.get('symbol')
        action = rec.get('action')
        confidence = rec.get('confidence', 0.5)
        target_allocation = rec.get('target_allocation_percent', 5)
        reason = rec.get('reason', 'Auto recommendation')

        # Only execute trades with confidence >= 60%
        if confidence < 0.6:
            continue

        # Get current price
        price_info = get_stock_price(symbol)
        if not price_info['success']:
            continue

        current_price = price_info['price']

        # Calculate trade amount
        if action == "BUY":
            amount = portfolio['cash'] * (target_allocation / 100)
            if amount < 10:  # Skip tiny trades
                continue
        elif action == "SELL":
            if symbol not in portfolio['positions']:
                continue
            pos = portfolio['positions'][symbol]
            amount = pos['shares'] * current_price * (target_allocation / 100)
        else:
            continue

        # Execute the trade
        result = execute_paper_trade(
            symbol,
            action,
            amount,
            current_price,
            f"[AUTO] {reason}",
            confidence
        )

        if result['success']:
            executed_trades.append({
                'symbol': symbol,
                'action': action,
                'amount': amount,
                'price': current_price
            })

            # Track recommendation for accuracy
            st.session_state.recommendation_history.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': action,
                'entry_price': current_price,
                'confidence': confidence,
                'reason': reason,
                'auto_executed': True
            })

    # Save after all auto trades
    if executed_trades:
        save_portfolio()

    return executed_trades


def calculate_recommendation_accuracy():
    """Calculate accuracy of AI recommendations."""
    history = st.session_state.recommendation_history
    if not history:
        return None

    correct = 0
    total = 0

    for rec in history:
        if 'outcome' in rec:
            total += 1
            if rec['outcome'] == 'correct':
                correct += 1

    if total == 0:
        return None

    return {
        'accuracy': correct / total * 100,
        'correct': correct,
        'total': total
    }


def display_news(news_items):
    """Display news items."""
    if not news_items:
        st.info("No news items available.")
        return

    for news in news_items[:10]:
        with st.container():
            st.markdown(f"**{news['title']}**")
            st.caption(f"{news['category']} | {news.get('published', 'N/A')}")
            if news.get('summary'):
                st.write(news['summary'][:300] + "..." if len(news['summary']) > 300 else news['summary'])
            st.divider()


def main():
    initialize_portfolio()

    # Header
    st.markdown("<h1 class='main-header'>📈 Auto Trading Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>News-Based Automated Paper Trading</p>", unsafe_allow_html=True)

    # Market Status Banner
    market_open, market_status = is_market_open()
    now_et = datetime.now(EASTERN_TZ)

    # Auto-refresh every 30 seconds (always active)
    st_autorefresh(interval=30 * 1000, key="auto_refresh")

    # Top-right loading indicator placeholder (fixed position via CSS)
    loading_indicator = st.empty()

    if market_open:
        st.success(f"🟢 {market_status} | {now_et.strftime('%I:%M %p ET')} | Auto-trading active | Refreshing every 30s")
    else:
        st.warning(f"🔴 {market_status} | Current time: {now_et.strftime('%I:%M %p ET, %A')} | Trading paused | Refreshing every 30s")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Analysis Engine Status
        st.subheader("Analysis Engine")
        st.success("Local NLP Engine (no API needed)")
        st.caption("Sentiment: Financial lexicon | Technical: RSI, MACD, SMA/EMA, Volume | Companies: Entity detection")

        st.divider()

        # Portfolio Summary
        st.subheader("💼 Portfolio Summary")
        portfolio = st.session_state.portfolio
        current_value = calculate_portfolio_value()
        total_return = current_value - portfolio['initial_capital']
        return_pct = (total_return / portfolio['initial_capital']) * 100

        st.metric("Portfolio Value", f"${current_value:,.2f}", f"{return_pct:+.2f}%")
        st.metric("Cash Available", f"${portfolio['cash']:,.2f}")
        st.metric("Total Trades", len(portfolio['trade_history']))

        # Accuracy
        accuracy = calculate_recommendation_accuracy()
        if accuracy:
            st.metric("Accuracy", f"{accuracy['accuracy']:.1f}%", f"{accuracy['correct']}/{accuracy['total']} correct")

        st.divider()

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            # Mark that we need to check for auto-trading
            st.session_state.force_news_check = True
            st.rerun()

        if st.button("🗑️ Reset Portfolio"):
            st.session_state.portfolio = {
                'cash': INITIAL_CAPITAL,
                'positions': {},
                'trade_history': [],
                'initial_capital': INITIAL_CAPITAL,
                'created_at': datetime.now().isoformat()
            }
            st.session_state.recommendation_history = []
            st.session_state.last_news_hash = None
            if 'current_recommendations' in st.session_state:
                del st.session_state.current_recommendations
            # Delete saved file
            if os.path.exists(PORTFOLIO_FILE):
                os.remove(PORTFOLIO_FILE)
            st.rerun()

    # Main Content Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📰 News & Analysis", "📊 Paper Trading", "📈 Portfolio", "📜 Trade History", "🧠 Engine Thinking"])

    with tab1:
        st.header("Latest Market News")

        # Show spinning gear while fetching and analyzing
        loading_indicator.markdown(
            '<div class="loading-indicator"><div class="spinner-icon"></div>Updating...</div>',
            unsafe_allow_html=True,
        )

        data = fetch_news(["Top News", "Market News", "Stock Market"])

        # Detect news changes
        current_news_hash = get_news_hash(data['news'])
        previous_news_hash = st.session_state.get('last_news_hash')
        news_changed = previous_news_hash is None or previous_news_hash != current_news_hash
        st.session_state.last_news_hash = current_news_hash
        save_portfolio()

        # Run analysis (gear icon stays visible at top-right while this runs)
        market_open, _ = is_market_open()
        recommendations = generate_trade_recommendations(data['news'])

        # Clear loading gear
        loading_indicator.empty()

        st.caption(f"Last updated: {data['timestamp']} | {data['news_count']} articles")

        if recommendations.get('success'):
            st.session_state.current_recommendations = recommendations

            # Only execute trades during market hours AND when news changed
            if market_open and news_changed:
                st.session_state.force_news_check = False
                executed = auto_execute_trades(recommendations)
                if executed:
                    st.toast(f"Auto-traded {len(executed)} position(s)")
        else:
            st.warning(f"Analysis failed: {recommendations.get('error')}")

        display_news(data['news'])

        # ---------------------------------------------------------------
        # LIVE ANALYSIS WINDOW — shows engine thinking & stock ratings
        # Works anytime, even outside market hours
        # ---------------------------------------------------------------
        st.divider()
        st.header("Live Analysis")

        if 'current_recommendations' not in st.session_state:
            st.info("No analysis yet — click Refresh Data or wait for the next news cycle.")
        else:
            recs = st.session_state.current_recommendations

            st.markdown(f"**Summary:** {recs.get('analysis_summary', 'N/A')}")
            st.caption(f"Generated: {recs.get('timestamp', 'N/A')} | Stories: {recs.get('stories_analyzed', '?')} | Tickers: {recs.get('tickers_detected', '?')}")

            if not market_open:
                st.warning("Market is closed — showing analysis preview. Trades will execute automatically when the market opens.")

            # --- Stock Ratings Table ---
            st.subheader("Stock Ratings")
            for rec in recs.get('recommendations', []):
                action = rec['action']
                if action == "BUY":
                    color = "🟢"
                elif action == "SELL":
                    color = "🔴"
                else:
                    color = "⚪"

                with st.container():
                    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 2.5])

                    with col1:
                        st.markdown(f"### {color} {action} {rec['symbol']}")
                    with col2:
                        st.metric("Confidence", f"{rec['confidence']*100:.0f}%")
                    with col3:
                        st.metric("Allocation", f"{rec['target_allocation_percent']}%")
                    with col4:
                        score_bar = rec.get('aggregated_score', 0)
                        direction = "Bullish" if score_bar > 0 else "Bearish" if score_bar < 0 else "Neutral"
                        sent_score = rec.get('sentiment_score')
                        tech_score = rec.get('technical_score')
                        score_parts = f"Blended: {score_bar:+.4f} ({direction})"
                        if sent_score is not None:
                            score_parts += f" | Sentiment: {sent_score:+.4f}"
                        if tech_score is not None:
                            score_parts += f" | Technical: {tech_score:+.4f}"
                        st.caption(f"{score_parts} | From {rec.get('story_count', '?')} stories")
                        st.progress(min(1.0, abs(score_bar) * 3))  # visual bar

                    # Technical indicators expander
                    ta = rec.get('technical_indicators')
                    if ta:
                        with st.expander(f"Technical Indicators — {rec['symbol']}", expanded=False):
                            tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                            with tc1:
                                rsi_val = ta.get('rsi', 0)
                                rsi_label = "Oversold" if rsi_val < 30 else "Overbought" if rsi_val > 70 else "Neutral"
                                rsi_icon = "🟢" if rsi_val < 40 else "🔴" if rsi_val > 60 else "⚪"
                                st.metric("RSI (14)", f"{rsi_val:.1f}")
                                st.caption(f"{rsi_icon} {rsi_label}")
                            with tc2:
                                macd_val = ta.get('macd', 0)
                                macd_hist = ta.get('macd_histogram', 0)
                                if ta.get('macd_bullish_crossover'):
                                    macd_status = "🟢 Bullish Cross"
                                elif ta.get('macd_bearish_crossover'):
                                    macd_status = "🔴 Bearish Cross"
                                else:
                                    macd_status = "🟢 Positive" if macd_hist > 0 else "🔴 Negative"
                                st.metric("MACD", f"{macd_val:.4f}")
                                st.caption(macd_status)
                            with tc3:
                                if ta.get('sma20_above_sma50'):
                                    trend = "🟢 Golden Cross"
                                else:
                                    trend = "🔴 Death Cross"
                                sma20 = ta.get('sma_20', 0)
                                sma50 = ta.get('sma_50')
                                st.metric("SMA 20", f"${sma20:.2f}")
                                if sma50 is not None:
                                    st.caption(f"SMA 50: ${sma50:.2f} | {trend}")
                                else:
                                    st.caption("SMA 50: N/A")
                            with tc4:
                                roc_val = ta.get('roc')
                                if roc_val is not None:
                                    roc_icon = "🟢" if roc_val > 2 else "🔴" if roc_val < -2 else "⚪"
                                    roc_label = "Rising" if roc_val > 0 else "Falling"
                                    st.metric("ROC (10d)", f"{roc_val:+.2f}%")
                                    st.caption(f"{roc_icon} {roc_label}")
                                else:
                                    st.metric("ROC (10d)", "N/A")
                            with tc5:
                                vol_ratio = ta.get('volume_ratio', 1.0)
                                vol_label = "High" if vol_ratio > 1.5 else "Low" if vol_ratio < 0.5 else "Normal"
                                vol_icon = "🔵" if vol_ratio > 1.5 else "⚪" if vol_ratio < 0.5 else "🟢"
                                st.metric("Volume", f"{vol_ratio:.1f}x avg")
                                st.caption(f"{vol_icon} {vol_label}")

                    st.divider()

            if not recs.get('recommendations'):
                st.info("No actionable signals found in current news.")

    with tab2:
        st.header("📊 Execute Paper Trades")

        market_open, market_status = is_market_open()

        if not market_open:
            st.warning(f"⚠️ {market_status}")
            st.info("Trades will be queued and can only execute when the market is open (Mon-Fri, 9:30 AM - 4:00 PM ET)")

        st.caption("Trades are executed automatically when news updates during market hours.")

        # Manual Trading Section
        st.subheader("Manual Trade")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            manual_symbol = st.selectbox("Symbol", TRADEABLE_STOCKS)

        with col2:
            manual_action = st.selectbox("Action", ["BUY", "SELL"])

        with col3:
            manual_amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0)

        with col4:
            st.write("")
            st.write("")
            if st.button("Execute Manual Trade", disabled=not market_open):
                price_info = get_stock_price(manual_symbol)
                if price_info['success']:
                    result = execute_paper_trade(
                        manual_symbol,
                        manual_action,
                        manual_amount,
                        price_info['price'],
                        "Manual trade",
                        1.0
                    )
                    if result['success']:
                        st.success(f"✅ {manual_action} {manual_symbol} executed at ${price_info['price']:.2f}")
                        st.rerun()
                    else:
                        st.error(result['error'])

    with tab3:
        st.header("📈 Portfolio Positions")

        portfolio = st.session_state.portfolio
        current_value = calculate_portfolio_value()
        total_return = current_value - portfolio['initial_capital']
        return_pct = (total_return / portfolio['initial_capital']) * 100

        # Portfolio metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Value", f"${current_value:,.2f}")
        with col2:
            delta_color = "normal" if total_return >= 0 else "inverse"
            st.metric("Total Return", f"${total_return:,.2f}", f"{return_pct:+.2f}%")
        with col3:
            st.metric("Cash", f"${portfolio['cash']:,.2f}")
        with col4:
            st.metric("Positions", len(portfolio['positions']))

        st.divider()

        # Current positions
        if portfolio['positions']:
            st.subheader("Current Holdings")

            positions_data = []
            symbols = list(portfolio['positions'].keys())
            prices = get_multiple_prices(symbols)

            for symbol, pos in portfolio['positions'].items():
                current_price = prices.get(symbol, pos['avg_cost'])
                pnl, pnl_pct = calculate_position_pnl(symbol, current_price)
                current_value_pos = pos['shares'] * current_price

                positions_data.append({
                    'Symbol': symbol,
                    'Shares': f"{pos['shares']:.4f}",
                    'Avg Cost': f"${pos['avg_cost']:.2f}",
                    'Current Price': f"${current_price:.2f}",
                    'Value': f"${current_value_pos:,.2f}",
                    'P&L': f"${pnl:,.2f}",
                    'P&L %': f"{pnl_pct:+.2f}%"
                })

            st.dataframe(positions_data, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions. Execute some trades to build your portfolio.")

        # Accuracy Section
        st.divider()
        st.subheader("🎯 Recommendation Accuracy")

        accuracy = calculate_recommendation_accuracy()
        if accuracy:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy Rate", f"{accuracy['accuracy']:.1f}%")
            with col2:
                st.metric("Correct Predictions", accuracy['correct'])
            with col3:
                st.metric("Total Evaluated", accuracy['total'])
        else:
            st.info("Accuracy tracking requires closed positions. Make some trades and close them to see accuracy metrics.")

    with tab4:
        st.header("📜 Trade History")

        portfolio = st.session_state.portfolio

        if portfolio['trade_history']:
            history_data = []
            for trade in reversed(portfolio['trade_history'][-50:]):  # Last 50 trades
                history_data.append({
                    'Time': trade['timestamp'][:19],
                    'Symbol': trade['symbol'],
                    'Action': trade['action'],
                    'Shares': f"{trade['shares']:.4f}",
                    'Price': f"${trade['price']:.2f}",
                    'Amount': f"${trade['amount']:,.2f}",
                    'Confidence': f"{trade['confidence']*100:.0f}%",
                    'Reason': trade['reason'][:50] + "..." if len(trade['reason']) > 50 else trade['reason']
                })

            st.dataframe(history_data, use_container_width=True, hide_index=True)

            # Download trade history
            st.download_button(
                "📥 Download Trade History",
                data=json.dumps(portfolio['trade_history'], indent=2),
                file_name=f"trade_history_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        else:
            st.info("No trades executed yet.")

    with tab5:
        st.header("🧠 Engine Thinking — Per-Story Breakdown")

        recs = st.session_state.get('current_recommendations', {})
        details = recs.get('story_details', [])

        if not details:
            st.info("No analysis data available yet. Wait for the next refresh cycle.")
        else:
            st.caption(f"{len(details)} stories analyzed | {recs.get('analysis_summary', '')}")

            for i, d in enumerate(details):
                sent = d['sentiment']
                if sent == "positive":
                    sent_icon = "🟢"
                elif sent == "negative":
                    sent_icon = "🔴"
                else:
                    sent_icon = "⚪"

                st.markdown(f"**Story {i+1}:** {d['snippet']}...")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"{sent_icon} **Sentiment:** {sent} ({d['sentiment_score']:+.4f})")
                    st.caption(f"Confidence: {d['sentiment_confidence']:.0%} | {d['word_hits']} keyword hits")
                with c2:
                    companies = d.get('companies_detected', {})
                    if companies:
                        tickers_str = ", ".join(f"**{t}** ({r:.0%})" for t, r in sorted(companies.items(), key=lambda x: -x[1]))
                        st.markdown(f"Companies: {tickers_str}")
                    else:
                        st.caption("No direct company mentions")
                with c3:
                    themes = d.get('themes_matched', {})
                    if themes:
                        themes_str = ", ".join(f"**{t}** ({r:.0%})" for t, r in sorted(themes.items(), key=lambda x: -x[1])[:6])
                        st.markdown(f"Themes: {themes_str}")
                    else:
                        st.caption("No theme matches")

                st.caption(f"Age: {d['age_hours']:.1f}h | Combined tickers: {', '.join(d.get('combined_tickers', {}).keys()) or 'none'}")
                st.divider()


if __name__ == "__main__":
    main()
