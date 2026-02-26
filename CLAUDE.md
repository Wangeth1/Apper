# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Auto Trading Analyzer — a Streamlit-based paper trading application that fetches Yahoo Finance news, analyzes sentiment, generates trade recommendations, and simulates trades with a $100k virtual portfolio.

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Environment setup: copy `.env.example` to `.env` and add API keys as needed.

## Architecture

There are two independent analysis paths:

### 1. Local NLP Path (primary, no API keys needed)
- **app.py** — Streamlit UI, portfolio management, news fetching, auto-refresh, paper trading loop
- **local_analyzer.py** — Self-contained rule-based trading engine with these internal components:
  - `SentimentAnalyzer` — financial lexicon-based sentiment scoring
  - `CompanyDetector` — entity extraction to ticker symbols (aliases + regex + dynamic resolution)
  - `DynamicResolver` — resolves any company name/ticker to valid US stock via `yf.Search()` with 30-min cache. Extracts proper nouns and ticker patterns from text automatically.
  - `ThemeMapper` — keyword themes to related stocks (sectors: AI, EV, rideshare, crypto, defense, etc.)
  - `ExchangePriceFilter` — validates US-listed stocks with price >= $8
  - `TechnicalAnalyzer` — RSI(14), SMA(20/50), EMA(12), MACD(12,26,9), ROC(10), Volume analysis via yfinance history. Scores each indicator to [-1,+1] and produces a composite technical score.
  - `SignalGenerator` — blends sentiment scores (55%) with technical scores (45%), applies time-decay aggregation → BUY/SELL/HOLD signals
  - `LocalTradingEngine` — orchestrator class used by `app.py`

### 2. LLM-powered Path (requires API keys)
- **stock_analyzer.py** — CLI tool that fetches Yahoo Finance news and uses an LLM to generate recommendations. Uses `llm_provider.py` for AI analysis.
- **llm_provider.py** — Multi-provider LLM abstraction layer (Gemini, OpenAI, Anthropic). Singleton pattern with factory. Configured via env vars: `LLM_PROVIDER`, `LLM_MODEL`, plus provider-specific API keys.

### Data Flow
`app.py` uses `LocalTradingEngine` from `local_analyzer.py` to analyze news → generate signals → execute paper trades. Portfolio state is persisted in `portfolio_data.json`. The LLM path (`stock_analyzer.py` + `llm_provider.py`) is a standalone CLI alternative.

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic Claude API access |
| `LLM_PROVIDER` | Provider selection: `gemini`, `openai`, `anthropic` (for stock_analyzer.py) |
| `LLM_MODEL` | Optional model override |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Google Gemini access |
| `OPENAI_API_KEY` | OpenAI access |
