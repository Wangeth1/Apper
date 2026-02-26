#!/usr/bin/env python3
"""
Self-contained local NLP trading decision engine.
Converts raw news text into structured trading recommendations
using only rule-based and local NLP methods — no external AI APIs.

Modules:
    1. SentimentAnalyzer   — financial-lexicon sentiment scoring
    2. CompanyDetector     — entity extraction → ticker symbols
    3. DynamicResolver     — yfinance Search-based ticker resolution
    4. ThemeMapper         — keyword themes → related stocks
    5. ExchangePriceFilter — validates US-listed, price >= $8
    6. TechnicalAnalyzer   — RSI, SMA/EMA, MACD, ROC, Volume indicators
    7. SignalGenerator     — blends sentiment + technical → BUY/SELL/HOLD
    8. LocalTradingEngine  — orchestrator
"""

import re
import math
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

import yfinance as yf

# ---------------------------------------------------------------------------
# 1. FINANCIAL SENTIMENT LEXICON
#    Each word maps to a sentiment score in [-1.0, +1.0].
#    Positive = bullish, Negative = bearish.
# ---------------------------------------------------------------------------

FINANCIAL_LEXICON: dict[str, float] = {
    # --- Strong Positive (0.6 – 1.0) ---
    "soar": 0.9, "soars": 0.9, "soared": 0.9, "soaring": 0.9,
    "surge": 0.85, "surges": 0.85, "surged": 0.85, "surging": 0.85,
    "skyrocket": 0.9, "skyrockets": 0.9, "skyrocketed": 0.9,
    "rally": 0.8, "rallies": 0.8, "rallied": 0.8, "rallying": 0.8,
    "boom": 0.8, "booming": 0.8, "booms": 0.8,
    "breakout": 0.75, "breakthrough": 0.8,
    "outperform": 0.7, "outperforms": 0.7, "outperformed": 0.7,
    "beat": 0.65, "beats": 0.65, "beating": 0.65,
    "exceed": 0.65, "exceeds": 0.65, "exceeded": 0.65, "exceeding": 0.65,
    "record": 0.6, "all-time high": 0.85, "bullish": 0.75,
    "upgrade": 0.7, "upgraded": 0.7, "upgrades": 0.7,
    "profit": 0.6, "profits": 0.6, "profitable": 0.65, "profitability": 0.6,
    "revenue growth": 0.7, "earnings beat": 0.75,
    "strong": 0.5, "strength": 0.5, "robust": 0.55,
    "optimistic": 0.6, "optimism": 0.6, "upbeat": 0.55,
    "momentum": 0.5, "accelerate": 0.6, "accelerates": 0.6, "accelerating": 0.6,
    "innovation": 0.5, "innovative": 0.5,
    "dividend": 0.4, "buyback": 0.5, "repurchase": 0.5,
    "expansion": 0.5, "expand": 0.45, "expands": 0.45, "expanding": 0.45,
    "recovery": 0.55, "recover": 0.5, "recovers": 0.5, "recovering": 0.5,
    "gain": 0.55, "gains": 0.55, "gained": 0.55, "gaining": 0.55,
    "jump": 0.6, "jumps": 0.6, "jumped": 0.6, "jumping": 0.6,
    "climb": 0.5, "climbs": 0.5, "climbed": 0.5, "climbing": 0.5,
    "rise": 0.5, "rises": 0.5, "rising": 0.5, "risen": 0.5,
    "up": 0.3, "higher": 0.35, "high": 0.25, "positive": 0.4,
    "growth": 0.5, "growing": 0.45, "grew": 0.45, "grow": 0.4,
    "boost": 0.55, "boosts": 0.55, "boosted": 0.55, "boosting": 0.55,
    "win": 0.5, "wins": 0.5, "winning": 0.5, "won": 0.5,
    "success": 0.55, "successful": 0.55,
    "demand": 0.4, "opportunity": 0.45, "opportunities": 0.45,
    "approval": 0.5, "approved": 0.55,
    "acquisition": 0.4, "acquire": 0.4, "acquires": 0.4, "acquired": 0.4,
    "merger": 0.35, "deal": 0.35, "partnership": 0.4,

    # --- Moderate Positive (0.2 – 0.5) ---
    "stable": 0.25, "stability": 0.25, "steady": 0.25,
    "resilient": 0.35, "resilience": 0.35,
    "improve": 0.4, "improves": 0.4, "improved": 0.4, "improving": 0.4,
    "increase": 0.35, "increases": 0.35, "increased": 0.35, "increasing": 0.35,
    "launch": 0.35, "launches": 0.35, "launched": 0.35,
    "invest": 0.3, "investment": 0.3, "investing": 0.3,
    "confident": 0.4, "confidence": 0.4,

    # --- Strong Negative (-0.6 to -1.0) ---
    "crash": -0.9, "crashes": -0.9, "crashed": -0.9, "crashing": -0.9,
    "plunge": -0.85, "plunges": -0.85, "plunged": -0.85, "plunging": -0.85,
    "collapse": -0.85, "collapses": -0.85, "collapsed": -0.85,
    "tank": -0.8, "tanks": -0.8, "tanked": -0.8, "tanking": -0.8,
    "tumble": -0.75, "tumbles": -0.75, "tumbled": -0.75, "tumbling": -0.75,
    "plummet": -0.85, "plummets": -0.85, "plummeted": -0.85,
    "selloff": -0.7, "sell-off": -0.7,
    "bearish": -0.75, "bear market": -0.8,
    "downgrade": -0.7, "downgraded": -0.7, "downgrades": -0.7,
    "bankruptcy": -0.95, "bankrupt": -0.95,
    "default": -0.8, "defaults": -0.8, "defaulted": -0.8,
    "recession": -0.75, "recessionary": -0.7,
    "layoff": -0.6, "layoffs": -0.65, "laid off": -0.6,
    "loss": -0.55, "losses": -0.55,
    "miss": -0.55, "misses": -0.55, "missed": -0.55, "missing": -0.4,
    "fraud": -0.9, "scandal": -0.8, "investigation": -0.5,
    "lawsuit": -0.55, "litigation": -0.5, "sued": -0.55,
    "fine": -0.45, "fined": -0.5, "penalty": -0.5, "penalties": -0.5,
    "debt": -0.35, "overvalued": -0.55,
    "decline": -0.5, "declines": -0.5, "declined": -0.5, "declining": -0.5,
    "drop": -0.5, "drops": -0.5, "dropped": -0.5, "dropping": -0.5,
    "fall": -0.5, "falls": -0.5, "fell": -0.5, "falling": -0.5,
    "sink": -0.6, "sinks": -0.6, "sank": -0.6, "sinking": -0.6,
    "slide": -0.5, "slides": -0.5, "slid": -0.5, "sliding": -0.5,
    "slump": -0.6, "slumps": -0.6, "slumped": -0.6, "slumping": -0.6,
    "weak": -0.45, "weakness": -0.45, "weaken": -0.45,
    "down": -0.3, "lower": -0.35, "low": -0.25, "negative": -0.4,
    "shrink": -0.5, "shrinks": -0.5, "shrank": -0.5, "shrinking": -0.5,
    "cut": -0.4, "cuts": -0.4, "cutting": -0.4,
    "risk": -0.3, "risks": -0.3, "risky": -0.35,
    "warning": -0.5, "warn": -0.5, "warns": -0.5, "warned": -0.5,
    "concern": -0.35, "concerns": -0.35, "concerned": -0.35,
    "fear": -0.45, "fears": -0.45, "fearful": -0.5,
    "uncertainty": -0.4, "uncertain": -0.4, "volatile": -0.35, "volatility": -0.35,
    "inflation": -0.35, "inflationary": -0.35,
    "tariff": -0.4, "tariffs": -0.4, "sanctions": -0.45,
    "shortage": -0.4, "shortages": -0.4,
    "delay": -0.35, "delayed": -0.35, "delays": -0.35,
    "recall": -0.5, "recalls": -0.5, "recalled": -0.5,
    "shutdown": -0.55, "closure": -0.5, "close": -0.2,
    "struggle": -0.45, "struggles": -0.45, "struggling": -0.45,
    "disappoint": -0.55, "disappoints": -0.55, "disappointed": -0.55, "disappointing": -0.55,
    "underperform": -0.55, "underperforms": -0.55, "underperformed": -0.55,

    # --- Neutral / Context-dependent ---
    "flat": -0.05, "unchanged": 0.0, "mixed": -0.05,
    "hold": 0.0, "maintain": 0.1, "maintains": 0.1,
    "report": 0.0, "reports": 0.0, "reported": 0.0,
    "announce": 0.1, "announces": 0.1, "announced": 0.1,
    "expect": 0.1, "expects": 0.1, "expected": 0.05,

    # --- Rate / Fed related ---
    "rate hike": -0.45, "rate cut": 0.5, "rate increase": -0.4,
    "hawkish": -0.4, "dovish": 0.4,
    "tighten": -0.35, "tightening": -0.35,
    "easing": 0.4, "stimulus": 0.45,
}

# Negation words that flip sentiment
NEGATION_WORDS = frozenset([
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "cannot", "can't", "won't", "don't",
    "doesn't", "didn't", "wasn't", "weren't", "isn't", "aren't",
    "wouldn't", "shouldn't", "couldn't", "hardly", "barely", "scarcely",
    "fail", "fails", "failed", "failing",
])

# Intensifiers that amplify sentiment
INTENSIFIERS = {
    "very": 1.3, "extremely": 1.5, "significantly": 1.4,
    "sharply": 1.4, "dramatically": 1.5, "massively": 1.5,
    "strongly": 1.3, "highly": 1.3, "deeply": 1.3,
    "substantially": 1.3, "considerably": 1.25,
    "slightly": 0.6, "marginally": 0.5, "somewhat": 0.7, "modestly": 0.7,
}

# ---------------------------------------------------------------------------
# 2. COMPANY NAME → TICKER MAPPING
#    Includes full names, common abbreviations, products, and key people.
# ---------------------------------------------------------------------------

COMPANY_ALIASES: dict[str, str] = {
    # AAPL
    "apple": "AAPL", "iphone": "AAPL", "ipad": "AAPL", "macbook": "AAPL",
    "mac": "AAPL", "tim cook": "AAPL", "app store": "AAPL", "apple vision": "AAPL",
    # MSFT
    "microsoft": "MSFT", "windows": "MSFT", "azure": "MSFT", "xbox": "MSFT",
    "satya nadella": "MSFT", "linkedin": "MSFT", "teams": "MSFT",
    "copilot": "MSFT", "bing": "MSFT", "openai": "MSFT",
    # GOOGL
    "google": "GOOGL", "alphabet": "GOOGL", "youtube": "GOOGL",
    "android": "GOOGL", "chrome": "GOOGL", "waymo": "GOOGL",
    "sundar pichai": "GOOGL", "deepmind": "GOOGL", "gemini ai": "GOOGL",
    # AMZN
    "amazon": "AMZN", "aws": "AMZN", "prime": "AMZN", "alexa": "AMZN",
    "andy jassy": "AMZN", "whole foods": "AMZN",
    # NVDA
    "nvidia": "NVDA", "geforce": "NVDA", "cuda": "NVDA",
    "jensen huang": "NVDA", "rtx": "NVDA",
    # META
    "meta": "META", "facebook": "META", "instagram": "META",
    "whatsapp": "META", "mark zuckerberg": "META", "zuckerberg": "META",
    "metaverse": "META", "threads app": "META",
    # TSLA
    "tesla": "TSLA", "elon musk": "TSLA", "musk": "TSLA",
    "cybertruck": "TSLA", "model 3": "TSLA", "model y": "TSLA",
    "autopilot": "TSLA", "supercharger": "TSLA", "spacex": "TSLA",
    # NFLX
    "netflix": "NFLX",
    # AMD
    "amd": "AMD", "advanced micro devices": "AMD", "radeon": "AMD",
    "ryzen": "AMD", "lisa su": "AMD", "epyc": "AMD",
    # INTC
    "intel": "INTC", "pat gelsinger": "INTC", "core ultra": "INTC",
    # PYPL
    "paypal": "PYPL", "venmo": "PYPL",
    # ADBE
    "adobe": "ADBE", "photoshop": "ADBE", "creative cloud": "ADBE",
    # CSCO
    "cisco": "CSCO", "webex": "CSCO",
    # CMCSA
    "comcast": "CMCSA", "nbcuniversal": "CMCSA", "nbc": "CMCSA", "peacock": "CMCSA",
    # PEP
    "pepsico": "PEP", "pepsi": "PEP", "frito-lay": "PEP", "gatorade": "PEP",
    # COST
    "costco": "COST",
    # TMUS
    "t-mobile": "TMUS",
    # AVGO
    "broadcom": "AVGO", "vmware": "AVGO",
    # TXN
    "texas instruments": "TXN",
    # QCOM
    "qualcomm": "QCOM", "snapdragon": "QCOM",
    # JPM
    "jpmorgan": "JPM", "jp morgan": "JPM", "jamie dimon": "JPM",
    "chase": "JPM", "j.p. morgan": "JPM",
    # V
    "visa": "V",
    # JNJ
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ", "j&j": "JNJ",
    # WMT
    "walmart": "WMT", "wal-mart": "WMT",
    # PG
    "procter & gamble": "PG", "procter and gamble": "PG", "p&g": "PG",
    # MA
    "mastercard": "MA",
    # UNH
    "unitedhealth": "UNH", "united health": "UNH", "unitedhealthcare": "UNH",
    # HD
    "home depot": "HD",
    # DIS
    "disney": "DIS", "walt disney": "DIS", "disney+": "DIS", "disney plus": "DIS",
    "hulu": "DIS", "espn": "DIS",
    # BAC
    "bank of america": "BAC",
    # XOM
    "exxon": "XOM", "exxonmobil": "XOM", "exxon mobil": "XOM",
    # CVX
    "chevron": "CVX",
    # KO
    "coca-cola": "KO", "coca cola": "KO", "coke": "KO",
    # PFE
    "pfizer": "PFE",
    # MRK
    "merck": "MRK", "keytruda": "MRK",
    # ABT
    "abbott": "ABT", "abbott labs": "ABT", "abbott laboratories": "ABT",
    # VZ
    "verizon": "VZ",
    # T
    "at&t": "T", "att": "T",
    # NKE
    "nike": "NKE", "jordan brand": "NKE",
    # MCD
    "mcdonald's": "MCD", "mcdonalds": "MCD", "mcdonald": "MCD",
}

# ---------------------------------------------------------------------------
# 3. THEME / SECTOR → TICKERS MAPPING
#    Keywords that imply relevance to a group of stocks.
#    Each entry: keyword → [(ticker, relevance_weight), ...]
# ---------------------------------------------------------------------------

THEME_MAP: dict[str, list[tuple[str, float]]] = {
    # Semiconductors & chips
    "semiconductor": [("NVDA", 1.0), ("AMD", 0.9), ("INTC", 0.85), ("AVGO", 0.8), ("TXN", 0.7), ("QCOM", 0.7)],
    "semiconductors": [("NVDA", 1.0), ("AMD", 0.9), ("INTC", 0.85), ("AVGO", 0.8), ("TXN", 0.7), ("QCOM", 0.7)],
    "chip": [("NVDA", 0.9), ("AMD", 0.85), ("INTC", 0.85), ("AVGO", 0.75), ("TXN", 0.7), ("QCOM", 0.7)],
    "chips": [("NVDA", 0.9), ("AMD", 0.85), ("INTC", 0.85), ("AVGO", 0.75), ("TXN", 0.7), ("QCOM", 0.7)],
    "chipmaker": [("NVDA", 0.95), ("AMD", 0.9), ("INTC", 0.9), ("AVGO", 0.8), ("TXN", 0.75), ("QCOM", 0.75)],
    "gpu": [("NVDA", 1.0), ("AMD", 0.8), ("INTC", 0.5)],
    "gpus": [("NVDA", 1.0), ("AMD", 0.8), ("INTC", 0.5)],
    "graphics card": [("NVDA", 1.0), ("AMD", 0.8)],
    "microchip": [("NVDA", 0.8), ("AMD", 0.8), ("INTC", 0.85), ("AVGO", 0.8), ("TXN", 0.8), ("QCOM", 0.75)],
    "microchips": [("NVDA", 0.8), ("AMD", 0.8), ("INTC", 0.85), ("AVGO", 0.8), ("TXN", 0.8), ("QCOM", 0.75)],
    "processor": [("INTC", 0.9), ("AMD", 0.9), ("QCOM", 0.7), ("NVDA", 0.6)],
    "processors": [("INTC", 0.9), ("AMD", 0.9), ("QCOM", 0.7), ("NVDA", 0.6)],
    "cpu": [("INTC", 0.95), ("AMD", 0.95)],
    "data center": [("NVDA", 0.9), ("AMD", 0.7), ("INTC", 0.6), ("MSFT", 0.5), ("AMZN", 0.5), ("GOOGL", 0.4)],

    # AI & machine learning
    "artificial intelligence": [("NVDA", 0.95), ("MSFT", 0.8), ("GOOGL", 0.8), ("META", 0.6), ("AMD", 0.5), ("AMZN", 0.5)],
    "ai": [("NVDA", 0.9), ("MSFT", 0.8), ("GOOGL", 0.8), ("META", 0.6), ("AMD", 0.5), ("AMZN", 0.5)],
    "machine learning": [("NVDA", 0.85), ("MSFT", 0.7), ("GOOGL", 0.75), ("META", 0.5), ("AMZN", 0.5)],
    "large language model": [("NVDA", 0.8), ("MSFT", 0.8), ("GOOGL", 0.8), ("META", 0.7)],
    "chatbot": [("MSFT", 0.7), ("GOOGL", 0.7), ("META", 0.5)],
    "generative ai": [("NVDA", 0.9), ("MSFT", 0.8), ("GOOGL", 0.8), ("META", 0.6)],

    # Cloud computing
    "cloud computing": [("AMZN", 0.9), ("MSFT", 0.9), ("GOOGL", 0.8)],
    "cloud": [("AMZN", 0.7), ("MSFT", 0.7), ("GOOGL", 0.6)],
    "saas": [("MSFT", 0.7), ("ADBE", 0.7), ("GOOGL", 0.5)],

    # Electric vehicles
    "electric vehicle": [("TSLA", 1.0)],
    "electric vehicles": [("TSLA", 1.0)],
    "ev": [("TSLA", 0.9)],
    "evs": [("TSLA", 0.9)],
    "battery": [("TSLA", 0.7)],
    "charging station": [("TSLA", 0.8)],
    "autonomous driving": [("TSLA", 0.8), ("GOOGL", 0.5)],
    "self-driving": [("TSLA", 0.8), ("GOOGL", 0.5)],

    # Social media & advertising
    "social media": [("META", 0.9), ("GOOGL", 0.5)],
    "digital advertising": [("META", 0.8), ("GOOGL", 0.85)],
    "online advertising": [("META", 0.8), ("GOOGL", 0.85)],
    "ad revenue": [("META", 0.85), ("GOOGL", 0.85)],

    # Streaming & entertainment
    "streaming": [("NFLX", 0.9), ("DIS", 0.7), ("AMZN", 0.4), ("CMCSA", 0.4)],
    "box office": [("DIS", 0.7), ("CMCSA", 0.5)],
    "subscriber": [("NFLX", 0.8), ("DIS", 0.5), ("TMUS", 0.4)],
    "subscribers": [("NFLX", 0.8), ("DIS", 0.5), ("TMUS", 0.4)],

    # E-commerce & retail
    "e-commerce": [("AMZN", 0.9), ("WMT", 0.5), ("COST", 0.3)],
    "ecommerce": [("AMZN", 0.9), ("WMT", 0.5), ("COST", 0.3)],
    "online shopping": [("AMZN", 0.85), ("WMT", 0.4)],
    "retail": [("WMT", 0.7), ("COST", 0.6), ("HD", 0.5), ("NKE", 0.4), ("MCD", 0.3)],
    "consumer spending": [("WMT", 0.6), ("COST", 0.5), ("HD", 0.5), ("MCD", 0.4), ("NKE", 0.4), ("PG", 0.4)],
    "black friday": [("AMZN", 0.7), ("WMT", 0.7), ("COST", 0.5), ("HD", 0.5)],

    # Payments & fintech
    "payment": [("V", 0.8), ("MA", 0.8), ("PYPL", 0.7)],
    "payments": [("V", 0.8), ("MA", 0.8), ("PYPL", 0.7)],
    "fintech": [("PYPL", 0.8), ("V", 0.6), ("MA", 0.6)],
    "credit card": [("V", 0.8), ("MA", 0.8), ("JPM", 0.5), ("BAC", 0.4)],
    "digital payment": [("V", 0.7), ("MA", 0.7), ("PYPL", 0.8)],

    # Banking & finance
    "banking": [("JPM", 0.9), ("BAC", 0.85)],
    "bank": [("JPM", 0.7), ("BAC", 0.7)],
    "interest rate": [("JPM", 0.7), ("BAC", 0.7), ("V", 0.3), ("MA", 0.3)],
    "interest rates": [("JPM", 0.7), ("BAC", 0.7), ("V", 0.3), ("MA", 0.3)],
    "federal reserve": [("JPM", 0.6), ("BAC", 0.6)],
    "fed": [("JPM", 0.5), ("BAC", 0.5)],
    "mortgage": [("JPM", 0.6), ("BAC", 0.6)],

    # Oil & energy
    "oil": [("XOM", 0.9), ("CVX", 0.9)],
    "crude oil": [("XOM", 0.95), ("CVX", 0.95)],
    "crude": [("XOM", 0.85), ("CVX", 0.85)],
    "natural gas": [("XOM", 0.7), ("CVX", 0.7)],
    "petroleum": [("XOM", 0.85), ("CVX", 0.85)],
    "opec": [("XOM", 0.8), ("CVX", 0.8)],
    "energy": [("XOM", 0.6), ("CVX", 0.6)],
    "oil price": [("XOM", 0.9), ("CVX", 0.9)],
    "gas price": [("XOM", 0.6), ("CVX", 0.6)],
    "drilling": [("XOM", 0.7), ("CVX", 0.7)],
    "refinery": [("XOM", 0.7), ("CVX", 0.7)],

    # Pharma & healthcare
    "pharmaceutical": [("PFE", 0.8), ("MRK", 0.8), ("JNJ", 0.7), ("ABT", 0.6)],
    "pharma": [("PFE", 0.8), ("MRK", 0.8), ("JNJ", 0.7), ("ABT", 0.6)],
    "drug": [("PFE", 0.7), ("MRK", 0.7), ("JNJ", 0.6)],
    "fda": [("PFE", 0.7), ("MRK", 0.7), ("JNJ", 0.6), ("ABT", 0.5)],
    "fda approval": [("PFE", 0.8), ("MRK", 0.8), ("JNJ", 0.7), ("ABT", 0.6)],
    "vaccine": [("PFE", 0.9), ("MRK", 0.6), ("JNJ", 0.7)],
    "clinical trial": [("PFE", 0.7), ("MRK", 0.7), ("JNJ", 0.6), ("ABT", 0.5)],
    "healthcare": [("UNH", 0.8), ("JNJ", 0.6), ("PFE", 0.5), ("MRK", 0.5), ("ABT", 0.6)],
    "health insurance": [("UNH", 0.9)],
    "medical device": [("ABT", 0.8), ("JNJ", 0.6)],

    # Telecom
    "5g": [("TMUS", 0.7), ("VZ", 0.7), ("T", 0.7), ("QCOM", 0.6)],
    "telecom": [("VZ", 0.7), ("T", 0.7), ("TMUS", 0.7), ("CSCO", 0.4)],
    "wireless": [("TMUS", 0.7), ("VZ", 0.7), ("T", 0.7)],
    "broadband": [("CMCSA", 0.7), ("VZ", 0.6), ("T", 0.6)],
    "network": [("CSCO", 0.6), ("VZ", 0.4), ("T", 0.4)],

    # Consumer packaged goods
    "consumer goods": [("PG", 0.7), ("KO", 0.5), ("PEP", 0.5)],
    "beverage": [("KO", 0.8), ("PEP", 0.8)],
    "beverages": [("KO", 0.8), ("PEP", 0.8)],
    "soft drink": [("KO", 0.8), ("PEP", 0.7)],
    "snack": [("PEP", 0.7)],
    "fast food": [("MCD", 0.9)],
    "restaurant": [("MCD", 0.7)],
    "sportswear": [("NKE", 0.9)],
    "athletic": [("NKE", 0.7)],
    "sneaker": [("NKE", 0.8)],
    "home improvement": [("HD", 0.9)],
    "housing": [("HD", 0.6)],

    # Cybersecurity
    "cybersecurity": [("CSCO", 0.6), ("MSFT", 0.4)],
    "data breach": [("CSCO", 0.4)],
    "hack": [("CSCO", 0.3)],

    # Trade war / geopolitics
    "trade war": [("AAPL", 0.5), ("NVDA", 0.5), ("AMD", 0.4), ("INTC", 0.4), ("AVGO", 0.4)],
    "china": [("AAPL", 0.4), ("NVDA", 0.5), ("INTC", 0.3), ("NKE", 0.3)],
    "supply chain": [("AAPL", 0.5), ("NVDA", 0.4), ("AMD", 0.3), ("WMT", 0.3)],

    # Rideshare / Mobility / Robotaxi
    "rideshare": [("TSLA", 0.4)],
    "ride-hailing": [("TSLA", 0.4)],
    "ride hailing": [("TSLA", 0.4)],
    "robotaxi": [("TSLA", 0.8), ("GOOGL", 0.7)],
    "robo taxi": [("TSLA", 0.8), ("GOOGL", 0.7)],
    "robo-taxi": [("TSLA", 0.8), ("GOOGL", 0.7)],
    "autonomous vehicle": [("TSLA", 0.85), ("GOOGL", 0.7)],
    "autonomous vehicles": [("TSLA", 0.85), ("GOOGL", 0.7)],
    "driverless": [("TSLA", 0.8), ("GOOGL", 0.75)],
    "food delivery": [("AMZN", 0.3)],
    "gig economy": [("PYPL", 0.3)],

    # Aerospace / Defense
    "defense": [("INTC", 0.3)],
    "defense contract": [("INTC", 0.3)],
    "military": [("INTC", 0.3)],
    "aerospace": [("INTC", 0.2)],
    "aircraft": [("INTC", 0.2)],

    # Cryptocurrency
    "cryptocurrency": [("PYPL", 0.3)],
    "crypto": [("PYPL", 0.3)],
    "bitcoin": [("PYPL", 0.3)],
    "blockchain": [("NVDA", 0.3)],

    # Travel / Hospitality
    "vacation rental": [("AMZN", 0.2)],
    "tourism": [("DIS", 0.4)],

    # EV expanded
    "electric truck": [("TSLA", 0.6)],
    "ev startup": [("TSLA", 0.4)],
    "luxury ev": [("TSLA", 0.5)],
    "electric pickup": [("TSLA", 0.4)],

    # Construction / Industrial
    "construction": [("HD", 0.4)],
    "heavy equipment": [("HD", 0.3)],
    "agriculture": [("COST", 0.2)],
    "infrastructure": [("HD", 0.3)],

    # Investment banking
    "investment bank": [("JPM", 0.7)],
    "investment banking": [("JPM", 0.7)],
    "wall street": [("JPM", 0.5)],
    "wealth management": [("JPM", 0.4)],

    # Online gaming / Betting
    "sports betting": [("DIS", 0.2)],
    "online gaming": [("MSFT", 0.4), ("NVDA", 0.4)],
    "video conferencing": [("MSFT", 0.5), ("GOOGL", 0.4)],
    "remote work": [("MSFT", 0.5)],
}


# ---------------------------------------------------------------------------
# 4. SENTIMENT ANALYZER
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """Scores text sentiment using the built-in financial lexicon."""

    def __init__(self, lexicon: Optional[dict[str, float]] = None):
        self._lexicon = lexicon or FINANCIAL_LEXICON
        # Pre-compile multi-word lexicon entries for matching
        self._multi_word = {k: v for k, v in self._lexicon.items() if " " in k or "-" in k}
        self._single_word = {k: v for k, v in self._lexicon.items() if " " not in k and "-" not in k}

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of a single text string.

        Returns:
            {
                "sentiment": "positive" | "negative" | "neutral",
                "score": float (-1.0 to 1.0),
                "confidence": float (0.0 to 1.0),
                "word_hits": int
            }
        """
        text_lower = text.lower()
        scores = []

        # 1) Multi-word phrase matching first
        for phrase, base_score in self._multi_word.items():
            count = text_lower.count(phrase)
            for _ in range(count):
                scores.append(base_score)

        # 2) Tokenize for single-word matching with negation/intensifier handling
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text_lower)

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Check for intensifier
            intensifier = INTENSIFIERS.get(token, 1.0)
            if intensifier != 1.0 and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token in self._single_word:
                    scores.append(self._single_word[next_token] * intensifier)
                    i += 2
                    continue

            # Check for negation
            if token in NEGATION_WORDS and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token in self._single_word:
                    scores.append(self._single_word[next_token] * -0.75)
                    i += 2
                    continue

            # Regular single-word match
            if token in self._single_word:
                scores.append(self._single_word[token])

            i += 1

        if not scores:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.1,
                "word_hits": 0,
            }

        avg_score = sum(scores) / len(scores)
        # Confidence is based on number of hits and score magnitude
        hit_confidence = min(1.0, len(scores) / 5.0)  # 5+ hits → max
        magnitude_confidence = min(1.0, abs(avg_score) * 2.0)
        confidence = (hit_confidence * 0.4) + (magnitude_confidence * 0.6)
        confidence = round(max(0.1, min(1.0, confidence)), 3)

        if avg_score > 0.05:
            label = "positive"
        elif avg_score < -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "sentiment": label,
            "score": round(avg_score, 4),
            "confidence": confidence,
            "word_hits": len(scores),
        }


# ---------------------------------------------------------------------------
# 5. COMPANY DETECTOR
# ---------------------------------------------------------------------------

class CompanyDetector:
    """Detects company references in text and maps them to ticker symbols."""

    # Tickers that are also common English words — require context
    AMBIGUOUS_TICKERS = frozenset(["T", "V", "HD"])

    def __init__(self, aliases: Optional[dict[str, str]] = None,
                 tradeable: Optional[list[str]] = None):
        self._aliases = aliases or COMPANY_ALIASES
        self._tradeable = set(tradeable) if tradeable else {
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
            "AMD", "INTC", "PYPL", "ADBE", "CSCO", "CMCSA", "PEP", "COST",
            "TMUS", "AVGO", "TXN", "QCOM", "JPM", "V", "JNJ", "WMT", "PG",
            "MA", "UNH", "HD", "DIS", "BAC", "XOM", "CVX", "KO", "PFE",
            "MRK", "ABT", "VZ", "T", "NKE", "MCD",
        }
        self._resolver = DynamicResolver()

    def detect(self, text: str) -> dict[str, float]:
        """
        Detect companies mentioned in text.

        Returns:
            dict of ticker → relevance score (0.0-1.0).
            Multiple mentions increase relevance.
        """
        text_lower = text.lower()
        ticker_hits: dict[str, int] = defaultdict(int)

        # 1) Alias matching (company names, products, people)
        for alias, ticker in self._aliases.items():
            if ticker not in self._tradeable:
                continue
            count = text_lower.count(alias)
            if count > 0:
                ticker_hits[ticker] += count

        # 2) Direct ticker symbol matching ($AAPL or standalone AAPL)
        #    Use word boundaries; skip ambiguous single-letter tickers
        for ticker in self._tradeable:
            if ticker in self.AMBIGUOUS_TICKERS:
                # Only match with $ prefix for ambiguous tickers
                pattern = rf'\${ticker}\b'
            else:
                pattern = rf'(?<![a-zA-Z]){ticker}(?![a-zA-Z])'
            matches = re.findall(pattern, text)
            if matches:
                ticker_hits[ticker] += len(matches)

        # 3) Dynamic resolution for unrecognized proper nouns and ticker patterns
        dynamic_hits = self._resolver.extract_and_resolve(text)
        for ticker, relevance in dynamic_hits.items():
            if ticker not in ticker_hits:
                # Store relevance as a pseudo hit count (scaled to compare with aliases)
                ticker_hits[ticker] = max(1, int(relevance * 3))

        # Convert hit counts to relevance scores (0-1)
        if not ticker_hits:
            return {}

        max_hits = max(ticker_hits.values())
        return {
            ticker: round(min(1.0, hits / max(max_hits, 1)), 3)
            for ticker, hits in ticker_hits.items()
        }


# ---------------------------------------------------------------------------
# 5b. DYNAMIC TICKER RESOLVER
#     Extracts proper nouns and ticker patterns from text, resolves them
#     to valid US-listed stock symbols via yfinance Search.
# ---------------------------------------------------------------------------

# Words that look like proper nouns but aren't company names
COMMON_PROPER_NOUNS = frozenset([
    # Days & months
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    # Financial/news terms commonly capitalized
    "Market", "Markets", "Stock", "Stocks", "Shares", "Share", "Trade", "Trading",
    "Report", "Revenue", "Quarter", "Quarterly", "Annual", "Earnings", "Profit",
    "Growth", "Sales", "Analyst", "Analysts", "Investors", "Investor",
    "Price", "Prices", "Rate", "Rates", "Index", "Funds", "Fund", "Bond", "Bonds",
    "Wall", "Street", "Exchange", "Global", "World", "International",
    "Today", "Yesterday", "Tomorrow", "According", "Reuters", "Bloomberg",
    # Directional / common sentence starters
    "The", "And", "For", "But", "Not", "All", "Can", "Was", "One", "Our",
    "Out", "Its", "Has", "His", "How", "New", "Now", "Old", "See", "Way",
    "Who", "Did", "Get", "Let", "Say", "She", "Too", "Use", "After", "Also",
    "Most", "Some", "What", "When", "With", "More", "From", "Over", "Into",
    "Just", "Than", "Very", "About", "Before", "Could", "Every", "First",
    "Major", "Other", "Since", "Their", "These", "Those", "Under", "Where",
    "While", "Would", "Should", "North", "South", "East", "West",
    "Chief", "President", "Chairman", "Board", "Company", "Companies",
    "Data", "Technology", "Technologies", "Capital", "Group", "Inc",
    "Corp", "Corporation", "Limited", "Partners", "Holdings",
    # Financial action/state words that get capitalized at sentence start
    "Rally", "Rallied", "Rallying", "Decline", "Declined", "Declining",
    "Surge", "Surged", "Surging", "Drop", "Dropped", "Dropping",
    "Rise", "Rising", "Risen", "Fall", "Falling", "Fell",
    "Gain", "Gained", "Gaining", "Loss", "Lost", "Losing",
    "Jump", "Jumped", "Jumping", "Slide", "Slid", "Sliding",
    "Boost", "Boosted", "Climb", "Climbed", "Plunge", "Plunged",
    "Beat", "Missed", "Exceeded", "Cut", "Raised", "Lower", "Higher",
    "Record", "Sell", "Buy", "Hold", "Yield", "Yields", "Chair",
    "Dovish", "Hawkish", "Bullish", "Bearish", "Soaring", "Soared",
    "Warning", "Alert", "Crisis", "Impact", "Shift", "Signal", "Signals",
    "Expected", "Announced", "Reported", "Filed", "Approved", "Denied",
    "Deal", "Merger", "Acquisition", "Partnership", "Launch", "Launched",
    "Federal", "Reserve", "Central", "Treasury", "Congress", "Senate",
    "Consensus", "Inflation", "Recession", "Unemployment", "Economy",
    "Sector", "Industry", "Regulatory", "Commission", "Authority",
])

# Uppercase sequences that aren't tickers
COMMON_UPPER_WORDS = frozenset([
    "US", "USA", "UK", "EU", "UN", "CEO", "CFO", "COO", "CTO", "CMO",
    "IPO", "GDP", "SEC", "FBI", "CIA", "NSA", "DOJ", "IRS", "EPA",
    "FDA", "FAA", "FCC", "FTC", "DOD", "NASA", "OPEC", "NATO",
    "ETF", "ESG", "GDP", "CPI", "PPI", "PMI", "PCE",
    "NYSE", "NASDAQ", "DJIA", "AI", "EV", "EVS", "IOT",
    "IT", "HR", "PR", "VP", "II", "III", "IV", "AM", "PM",
    "YOY", "QOQ", "MOM", "ATH", "ATL", "EPS", "PE", "ROI", "ROE",
    "YTD", "MTD", "QTD", "M&A", "R&D", "B2B", "B2C", "DC",
    "APR", "AUG", "DEC", "FEB", "JAN", "JUL", "JUN", "MAR",
    "NOV", "OCT", "SEP", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
])

_resolver_cache: dict[str, tuple[Optional[str], datetime]] = {}
_RESOLVER_CACHE_TTL = timedelta(minutes=30)


class DynamicResolver:
    """
    Resolves company names and ticker patterns to valid US-listed stock
    symbols using yfinance Search API with caching.
    """

    def resolve(self, name: str) -> Optional[str]:
        """
        Try to resolve a company name or ticker candidate to a valid US-listed ticker.
        Returns the ticker string or None.
        """
        cache_key = name.lower().strip()
        now = datetime.now()

        # Check cache
        if cache_key in _resolver_cache:
            cached_ticker, cached_at = _resolver_cache[cache_key]
            if (now - cached_at) < _RESOLVER_CACHE_TTL:
                return cached_ticker

        ticker = self._lookup(name)
        _resolver_cache[cache_key] = (ticker, now)
        return ticker

    @staticmethod
    def _lookup(name: str) -> Optional[str]:
        """Query yfinance Search for a company name, return first US-listed ticker."""
        try:
            from yfinance import Search
            results = Search(name)
            for quote in results.quotes[:3]:
                exchange = quote.get("exchange", "")
                if exchange in VALID_EXCHANGES:
                    symbol = quote.get("symbol", "")
                    if symbol:
                        return symbol
        except Exception:
            pass
        return None

    def extract_and_resolve(self, text: str) -> dict[str, float]:
        """
        Extract proper nouns and ticker-like patterns from text,
        resolve each to a valid US ticker via yfinance Search.

        Returns:
            dict of ticker → relevance (0.6-0.9)
        """
        candidates: dict[str, float] = {}  # search_term → relevance

        # 1) Proper nouns: capitalized words (2+ chars)
        for match in re.finditer(r'\b([A-Z][a-z]{1,20})\b', text):
            word = match.group(1)
            if word not in COMMON_PROPER_NOUNS and len(word) >= 3:
                candidates[word] = 0.7

        # 2) Uppercase ticker patterns: 2-5 uppercase letters
        for match in re.finditer(r'(?<![a-zA-Z])([A-Z]{2,5})(?![a-zA-Z])', text):
            word = match.group(1)
            if word not in COMMON_UPPER_WORDS:
                candidates[word] = 0.9

        # 3) Resolve each candidate
        resolved: dict[str, float] = {}
        for candidate, relevance in candidates.items():
            ticker = self.resolve(candidate)
            if ticker:
                # Keep highest relevance if same ticker found multiple ways
                resolved[ticker] = max(resolved.get(ticker, 0), relevance)

        return resolved


# ---------------------------------------------------------------------------
# 6. THEME MAPPER
# ---------------------------------------------------------------------------

class ThemeMapper:
    """Maps thematic keywords in text to related stock tickers."""

    def __init__(self, theme_map: Optional[dict] = None,
                 tradeable: Optional[set[str]] = None):
        self._themes = theme_map or THEME_MAP
        self._tradeable = tradeable or {
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
            "AMD", "INTC", "PYPL", "ADBE", "CSCO", "CMCSA", "PEP", "COST",
            "TMUS", "AVGO", "TXN", "QCOM", "JPM", "V", "JNJ", "WMT", "PG",
            "MA", "UNH", "HD", "DIS", "BAC", "XOM", "CVX", "KO", "PFE",
            "MRK", "ABT", "VZ", "T", "NKE", "MCD",
        }

    def map_themes(self, text: str) -> dict[str, float]:
        """
        Find thematic keywords in text and return implied tickers.

        Returns:
            dict of ticker → max relevance weight from matched themes.
        """
        text_lower = text.lower()
        ticker_relevance: dict[str, float] = defaultdict(float)

        for keyword, stock_list in self._themes.items():
            if keyword in text_lower:
                for ticker, weight in stock_list:
                    if ticker in self._tradeable:
                        # Keep the highest relevance seen
                        ticker_relevance[ticker] = max(ticker_relevance[ticker], weight)

        return dict(ticker_relevance)


# ---------------------------------------------------------------------------
# 7. EXCHANGE & PRICE FILTER
# ---------------------------------------------------------------------------

# Accepted exchange identifiers from yfinance
VALID_EXCHANGES = frozenset([
    "NMS",   # NASDAQ Global Select Market
    "NGM",   # NASDAQ Global Market
    "NCM",   # NASDAQ Capital Market
    "NYQ",   # NYSE
    "NYSE",
    "NASDAQ",
    "NasijdaqGS",  # variant strings yfinance sometimes returns
    "PCX",   # NYSE Arca
])

# Price cache to reduce yfinance calls within a single analysis run
_price_cache: dict[str, tuple[float, datetime]] = {}
_CACHE_TTL = timedelta(minutes=2)


class ExchangePriceFilter:
    """Filters stocks to ensure they are US-listed and above the minimum price."""

    MIN_PRICE = 8.0

    def __init__(self, min_price: float = 8.0):
        self.MIN_PRICE = min_price

    def get_price_info(self, symbol: str) -> dict:
        """
        Fetch price and exchange for a symbol.
        Uses a short-lived cache to avoid redundant API calls.
        """
        now = datetime.now()
        if symbol in _price_cache:
            price, cached_at = _price_cache[symbol]
            if now - cached_at < _CACHE_TTL:
                return {"symbol": symbol, "price": price, "success": True}

        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            exchange = info.get("exchange", "")

            if price is None:
                return {"symbol": symbol, "success": False, "reason": "no price data"}

            price = round(float(price), 2)
            _price_cache[symbol] = (price, now)

            return {
                "symbol": symbol,
                "price": price,
                "exchange": exchange,
                "success": True,
            }
        except Exception as e:
            return {"symbol": symbol, "success": False, "reason": str(e)}

    def filter(self, tickers: list[str]) -> dict[str, float]:
        """
        Filter a list of tickers. Returns {ticker: price} for those that pass.
        Filters out: price < MIN_PRICE and non-US exchanges.
        """
        valid = {}
        for symbol in tickers:
            info = self.get_price_info(symbol)
            if not info["success"]:
                continue
            if info["price"] < self.MIN_PRICE:
                continue
            valid[symbol] = info["price"]
        return valid


# ---------------------------------------------------------------------------
# 8. TECHNICAL ANALYZER
#    Computes RSI, SMA/EMA, MACD, Volume indicators from yfinance history.
#    Each indicator is scored to [-1, +1] and combined into a composite score.
# ---------------------------------------------------------------------------

_ta_cache: dict[str, tuple[dict, datetime]] = {}
_TA_CACHE_TTL = timedelta(minutes=10)


class TechnicalAnalyzer:
    """
    Fetches historical price data via yfinance and computes technical
    indicators per ticker. Returns a composite score in [-1, +1].
    """

    # Indicator weights for composite score
    RSI_WEIGHT = 0.25
    MA_WEIGHT = 0.30
    MACD_WEIGHT = 0.30
    ROC_WEIGHT = 0.15

    def analyze(self, symbol: str) -> dict:
        """
        Compute technical indicators for a single ticker.

        Returns:
            {
                "symbol": str,
                "technical_score": float (-1 to +1),
                "technical_confidence": float (0 to 1),
                "rsi": float,
                "rsi_score": float,
                "ma_score": float,
                "macd_score": float,
                "volume_ratio": float,
                "indicators": dict (full details),
                "success": bool,
            }
        """
        # Check cache
        now = datetime.now()
        if symbol in _ta_cache:
            cached, cached_at = _ta_cache[symbol]
            if (now - cached_at) < _TA_CACHE_TTL:
                return cached

        result = self._compute(symbol)
        _ta_cache[symbol] = (result, now)
        return result

    def _compute(self, symbol: str) -> dict:
        """Fetch history and compute all indicators."""
        fail = {"symbol": symbol, "success": False}

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo", interval="1d")
        except Exception:
            return fail

        if hist is None or len(hist) < 20:
            return fail

        closes = hist["Close"]
        volumes = hist["Volume"]

        # --- Compute each indicator ---
        rsi = self._compute_rsi(closes)
        ma = self._compute_ma_signals(closes)
        macd = self._compute_macd(closes)
        vol = self._compute_volume_signal(volumes)
        roc = self._compute_roc(closes)

        if rsi is None or ma is None or macd is None:
            return fail

        # --- Score each indicator ---
        rsi_score = self._score_rsi(rsi)
        ma_score = self._score_ma(ma)
        macd_score = self._score_macd(macd)
        roc_score = self._score_roc(roc) if roc is not None else 0.0

        # Volume: graduated multiplier based on how far above/below average
        vol_ratio = vol.get("volume_ratio", 1.0)
        if vol_ratio > 1.5:
            vol_multiplier = 1.0 + min(0.4, (vol_ratio - 1.0) * 0.2)
        elif vol_ratio < 0.5:
            vol_multiplier = 0.7
        else:
            vol_multiplier = 1.0

        # Composite score
        raw_score = (
            rsi_score * self.RSI_WEIGHT
            + ma_score * self.MA_WEIGHT
            + macd_score * self.MACD_WEIGHT
            + roc_score * self.ROC_WEIGHT
        )
        technical_score = max(-1.0, min(1.0, raw_score * vol_multiplier))

        # Confidence: higher when indicators agree and signal is strong
        signs = [rsi_score, ma_score, macd_score, roc_score]
        agreement = sum(1 for s in signs if s * raw_score > 0) / len(signs)
        signal_strength = min(1.0, abs(raw_score) * 2.0)
        technical_confidence = min(1.0, 0.3 + (agreement * 0.4) + (signal_strength * 0.3))

        indicators = {
            "rsi": round(rsi, 2),
            "sma_20": round(ma["sma_20"], 2),
            "sma_50": round(ma["sma_50"], 2) if ma["sma_50"] is not None else None,
            "ema_12": round(ma["ema_12"], 2),
            "price": round(ma["price"], 2),
            "price_above_sma20": ma["price_above_sma20"],
            "price_above_sma50": ma["price_above_sma50"],
            "sma20_above_sma50": ma["sma20_above_sma50"],
            "macd": round(macd["macd"], 4),
            "macd_signal": round(macd["signal"], 4),
            "macd_histogram": round(macd["histogram"], 4),
            "macd_bullish_crossover": macd["bullish_crossover"],
            "macd_bearish_crossover": macd["bearish_crossover"],
            "roc": round(roc, 4) if roc is not None else None,
            "volume_ratio": round(vol_ratio, 2),
            "avg_volume": vol.get("avg_volume"),
            "current_volume": vol.get("current_volume"),
        }

        return {
            "symbol": symbol,
            "technical_score": round(technical_score, 4),
            "technical_confidence": round(technical_confidence, 3),
            "rsi": round(rsi, 2),
            "rsi_score": round(rsi_score, 4),
            "ma_score": round(ma_score, 4),
            "macd_score": round(macd_score, 4),
            "roc_score": round(roc_score, 4),
            "volume_ratio": round(vol_ratio, 2),
            "indicators": indicators,
            "success": True,
        }

    # --- Indicator computations ---

    @staticmethod
    def _compute_rsi(closes, period: int = 14) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        last_gain = gain.iloc[-1]
        last_loss = loss.iloc[-1]
        if last_loss < 1e-10:
            return 100.0
        rs = last_gain / last_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _compute_ma_signals(closes) -> Optional[dict]:
        if len(closes) < 20:
            return None
        sma_20 = closes.rolling(20).mean().iloc[-1]
        sma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else None
        ema_12 = closes.ewm(span=12, adjust=False).mean().iloc[-1]
        current = closes.iloc[-1]
        return {
            "sma_20": sma_20,
            "sma_50": sma_50,
            "ema_12": ema_12,
            "price": current,
            "price_above_sma20": current > sma_20,
            "price_above_sma50": current > sma_50 if sma_50 is not None else True,
            "sma20_above_sma50": sma_20 > sma_50 if sma_50 is not None else True,
        }

    @staticmethod
    def _compute_macd(closes) -> Optional[dict]:
        if len(closes) < 26:
            return None
        ema_12 = closes.ewm(span=12, adjust=False).mean()
        ema_26 = closes.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        bullish_cross = (
            len(macd_line) >= 2
            and macd_line.iloc[-1] > signal_line.iloc[-1]
            and macd_line.iloc[-2] <= signal_line.iloc[-2]
        )
        bearish_cross = (
            len(macd_line) >= 2
            and macd_line.iloc[-1] < signal_line.iloc[-1]
            and macd_line.iloc[-2] >= signal_line.iloc[-2]
        )

        return {
            "macd": macd_line.iloc[-1],
            "signal": signal_line.iloc[-1],
            "histogram": histogram.iloc[-1],
            "bullish_crossover": bullish_cross,
            "bearish_crossover": bearish_cross,
        }

    @staticmethod
    def _compute_volume_signal(volumes) -> dict:
        if len(volumes) < 20:
            return {"volume_ratio": 1.0, "avg_volume": 0, "current_volume": 0}
        avg_vol = volumes.rolling(20).mean().iloc[-1]
        current_vol = volumes.iloc[-1]
        ratio = current_vol / max(avg_vol, 1)
        return {
            "volume_ratio": ratio,
            "avg_volume": avg_vol,
            "current_volume": current_vol,
        }

    @staticmethod
    def _compute_roc(closes, period: int = 10) -> Optional[float]:
        """Price Rate of Change over `period` days, as a percentage."""
        if len(closes) < period + 1:
            return None
        current = closes.iloc[-1]
        past = closes.iloc[-(period + 1)]
        if abs(past) < 1e-10:
            return None
        return ((current - past) / past) * 100.0

    # --- Scoring functions (each returns -1 to +1) ---

    @staticmethod
    def _score_rsi(rsi: float) -> float:
        """Smooth continuous RSI scoring."""
        if rsi <= 25:
            return 0.8 + 0.2 * (25 - rsi) / 25.0
        elif rsi <= 40:
            return 0.8 * (40 - rsi) / 15.0
        elif rsi <= 60:
            return 0.0
        elif rsi <= 75:
            return -0.8 * (rsi - 60) / 15.0
        else:
            return -0.8 - 0.2 * (rsi - 75) / 25.0

    @staticmethod
    def _score_roc(roc: float) -> float:
        """Score ROC using tanh for smooth sigmoid mapping."""
        return max(-1.0, min(1.0, math.tanh(roc / 8.0)))

    @staticmethod
    def _score_ma(ma: dict) -> float:
        score = 0.0
        score += 0.3 if ma["price_above_sma20"] else -0.3
        score += 0.3 if ma["price_above_sma50"] else -0.3
        score += 0.4 if ma["sma20_above_sma50"] else -0.4
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _score_macd(macd: dict) -> float:
        if macd["bullish_crossover"]:
            return 0.7
        if macd["bearish_crossover"]:
            return -0.7
        return 0.3 if macd["histogram"] > 0 else -0.3


# ---------------------------------------------------------------------------
# 9. SIGNAL GENERATOR
# ---------------------------------------------------------------------------

class SignalGenerator:
    """
    Aggregates per-story sentiment into per-stock trading signals.
    Applies time-decay so recent stories carry more weight.
    """

    # Decay half-life in hours (stories lose half their influence every N hours)
    DECAY_HALF_LIFE_HOURS = 6.0

    # Thresholds for converting aggregated score to action
    BUY_THRESHOLD = 0.15
    SELL_THRESHOLD = -0.15

    # Minimum confidence to produce a non-HOLD recommendation
    MIN_CONFIDENCE = 0.3

    def __init__(self, decay_half_life: float = 6.0):
        self.DECAY_HALF_LIFE_HOURS = decay_half_life
        self._lambda = math.log(2) / max(decay_half_life, 0.1)

    def _time_decay_weight(self, story_age_hours: float) -> float:
        """Exponential decay: weight = e^(-lambda * age)."""
        return math.exp(-self._lambda * max(story_age_hours, 0))

    def _position_size(self, confidence: float, abs_score: float) -> int:
        """
        Map confidence × score magnitude to a portfolio allocation % (1-20).
        """
        raw = confidence * abs_score * 30  # scale factor
        return max(1, min(20, int(round(raw))))

    # Blending weights for sentiment vs technical scores
    SENTIMENT_WEIGHT = 0.55
    TECHNICAL_WEIGHT = 0.45

    def generate(
        self,
        story_analyses: list[dict],
        now: Optional[datetime] = None,
        technical_scores: Optional[dict] = None,
    ) -> list[dict]:
        """
        Generate trading signals from analyzed stories, optionally blended
        with technical indicator scores.

        Args:
            story_analyses: list of dicts, each with:
                - "sentiment_score": float
                - "sentiment_confidence": float
                - "tickers": dict[ticker → relevance (0-1)]
                - "story_age_hours": float (0 = brand new)
            now: reference time (default: datetime.now())
            technical_scores: optional dict of ticker → TechnicalAnalyzer result

        Returns:
            list of recommendation dicts
        """
        # Aggregate weighted sentiment per ticker
        ticker_data: dict[str, dict] = defaultdict(lambda: {
            "weighted_score_sum": 0.0,
            "weight_sum": 0.0,
            "confidence_sum": 0.0,
            "story_count": 0,
            "reasons": [],
        })

        for analysis in story_analyses:
            sentiment_score = analysis.get("sentiment_score", 0.0)
            sentiment_conf = analysis.get("sentiment_confidence", 0.5)
            tickers = analysis.get("tickers", {})
            age_hours = analysis.get("story_age_hours", 0.0)
            story_text = analysis.get("story_snippet", "")

            decay_w = self._time_decay_weight(age_hours)

            for ticker, relevance in tickers.items():
                combined_weight = decay_w * relevance * sentiment_conf
                td = ticker_data[ticker]
                td["weighted_score_sum"] += sentiment_score * combined_weight
                td["weight_sum"] += combined_weight
                td["confidence_sum"] += sentiment_conf * relevance
                td["story_count"] += 1
                if story_text:
                    td["reasons"].append(story_text[:80])

        # Convert to recommendations
        recommendations = []
        for ticker, td in ticker_data.items():
            if td["weight_sum"] < 0.01:
                continue

            sentiment_agg = td["weighted_score_sum"] / td["weight_sum"]
            avg_confidence = td["confidence_sum"] / td["story_count"]
            sentiment_confidence = min(1.0, avg_confidence * min(1.0, td["story_count"] / 3.0))

            # Blend with technical analysis if available
            ta = None
            if technical_scores and ticker in technical_scores:
                ta = technical_scores[ticker]
                ta_score = ta["technical_score"]
                ta_conf = ta["technical_confidence"]
                agg_score = (sentiment_agg * self.SENTIMENT_WEIGHT) + (ta_score * self.TECHNICAL_WEIGHT)
                combined_confidence = (
                    sentiment_confidence * self.SENTIMENT_WEIGHT
                    + ta_conf * self.TECHNICAL_WEIGHT
                )
            else:
                agg_score = sentiment_agg
                combined_confidence = sentiment_confidence

            if agg_score > self.BUY_THRESHOLD and combined_confidence >= self.MIN_CONFIDENCE:
                action = "BUY"
            elif agg_score < self.SELL_THRESHOLD and combined_confidence >= self.MIN_CONFIDENCE:
                action = "SELL"
            else:
                action = "HOLD"

            reason = "; ".join(td["reasons"][:3])
            if len(td["reasons"]) > 3:
                reason += f" (+{len(td['reasons'])-3} more)"

            rec = {
                "symbol": ticker,
                "action": action,
                "confidence": round(combined_confidence, 3),
                "reason": reason,
                "target_allocation_percent": self._position_size(combined_confidence, abs(agg_score)),
                "aggregated_score": round(agg_score, 4),
                "story_count": td["story_count"],
                "sentiment_score": round(sentiment_agg, 4),
                "technical_score": round(ta["technical_score"], 4) if ta else None,
                "technical_indicators": ta.get("indicators") if ta else None,
            }
            recommendations.append(rec)

        # Sort: actionable (BUY/SELL) first, then by confidence descending
        action_order = {"BUY": 0, "SELL": 1, "HOLD": 2}
        recommendations.sort(key=lambda r: (action_order.get(r["action"], 3), -r["confidence"]))

        return recommendations


# ---------------------------------------------------------------------------
# 9. LOCAL TRADING ENGINE (Orchestrator)
# ---------------------------------------------------------------------------

class LocalTradingEngine:
    """
    End-to-end pipeline: raw news text → structured trading recommendations.

    Components wired together:
        SentimentAnalyzer → CompanyDetector → ThemeMapper
            → ExchangePriceFilter → TechnicalAnalyzer → SignalGenerator
    """

    def __init__(self, tradeable_stocks: Optional[list[str]] = None,
                 min_price: float = 8.0,
                 decay_half_life: float = 6.0):
        self._tradeable = set(tradeable_stocks) if tradeable_stocks else None
        self.sentiment = SentimentAnalyzer()
        self.companies = CompanyDetector(tradeable=self._tradeable)
        self.themes = ThemeMapper(tradeable=self._tradeable)
        self.price_filter = ExchangePriceFilter(min_price=min_price)
        self.technical = TechnicalAnalyzer()
        self.signals = SignalGenerator(decay_half_life=decay_half_life)

    def analyze_stories(
        self,
        stories: list[dict],
        now: Optional[datetime] = None,
        filter_hold: bool = True,
        max_recommendations: int = 8,
    ) -> dict:
        """
        Full pipeline: analyze a list of news stories and produce recommendations.

        Args:
            stories: list of dicts with at minimum:
                - "text": str (the raw story text — title + summary)
                Optional:
                - "published": str (ISO or RSS date string)
                - "title": str
                - "summary": str
            now: reference time for age calculation
            filter_hold: if True, omit HOLD recommendations from output
            max_recommendations: cap the number of recommendations

        Returns:
            {
                "success": True,
                "analysis_summary": str,
                "recommendations": [...],
                "timestamp": str,
                "provider": "local_nlp",
                "model": "financial_lexicon_v1",
                "stories_analyzed": int,
                "tickers_detected": int,
            }
        """
        if now is None:
            now = datetime.now()

        # --- Step 1: Analyze each story ---
        story_analyses = []
        all_tickers: dict[str, float] = defaultdict(float)

        story_details = []  # detailed per-story breakdown for UI

        for story in stories:
            text = story.get("text", "")
            if not text:
                # Build text from title + summary if "text" not provided
                title = story.get("title", "")
                summary = story.get("summary", "")
                text = f"{title}. {summary}".strip(". ")
            if not text:
                continue

            # Sentiment
            sent = self.sentiment.analyze(text)

            # Company detection (direct mentions)
            detected = self.companies.detect(text)

            # Theme mapping (inferred relevance)
            themed = self.themes.map_themes(text)

            # Merge: detected companies + themed companies
            combined_tickers: dict[str, float] = {}
            for t, rel in detected.items():
                combined_tickers[t] = rel
            for t, rel in themed.items():
                # Theme relevance is additive but capped
                combined_tickers[t] = min(1.0, combined_tickers.get(t, 0) + rel * 0.6)

            # Calculate story age
            age_hours = self._parse_age_hours(story.get("published", ""), now)

            # Save detailed breakdown for every story (even ones with no tickers)
            story_details.append({
                "snippet": text[:120],
                "sentiment": sent["sentiment"],
                "sentiment_score": sent["score"],
                "sentiment_confidence": sent["confidence"],
                "word_hits": sent["word_hits"],
                "companies_detected": dict(detected),
                "themes_matched": dict(themed),
                "combined_tickers": dict(combined_tickers),
                "age_hours": round(age_hours, 1),
            })

            if not combined_tickers:
                continue

            story_analyses.append({
                "sentiment_score": sent["score"],
                "sentiment_confidence": sent["confidence"],
                "tickers": combined_tickers,
                "story_age_hours": age_hours,
                "story_snippet": text[:100],
            })

            for t, rel in combined_tickers.items():
                all_tickers[t] = max(all_tickers[t], rel)

        if not story_analyses:
            return {
                "success": True,
                "analysis_summary": "No actionable stories found in the current news feed.",
                "recommendations": [],
                "story_details": story_details,
                "timestamp": now.isoformat(),
                "provider": "local_nlp",
                "model": "financial_lexicon_v1",
                "stories_analyzed": len(stories),
                "tickers_detected": 0,
            }

        # --- Step 2: Price-filter all candidate tickers ---
        candidate_tickers = list(all_tickers.keys())
        valid_prices = self.price_filter.filter(candidate_tickers)

        # Remove tickers that failed price filter from every analysis
        for analysis in story_analyses:
            analysis["tickers"] = {
                t: r for t, r in analysis["tickers"].items() if t in valid_prices
            }

        # Remove analyses with no remaining tickers
        story_analyses = [a for a in story_analyses if a["tickers"]]

        # --- Step 2.5: Technical analysis for all valid tickers ---
        technical_scores = {}
        for symbol in valid_prices:
            ta_result = self.technical.analyze(symbol)
            if ta_result.get("success"):
                technical_scores[symbol] = ta_result

        # --- Step 3: Generate signals (blended sentiment + technical) ---
        raw_recs = self.signals.generate(story_analyses, now, technical_scores=technical_scores)

        # Filter out HOLDs if requested
        if filter_hold:
            raw_recs = [r for r in raw_recs if r["action"] != "HOLD"]

        # Cap
        recommendations = raw_recs[:max_recommendations]

        # --- Step 4: Build summary ---
        buy_count = sum(1 for r in recommendations if r["action"] == "BUY")
        sell_count = sum(1 for r in recommendations if r["action"] == "SELL")

        if buy_count > sell_count:
            bias = "bullish"
        elif sell_count > buy_count:
            bias = "bearish"
        else:
            bias = "mixed"

        ta_note = f" Technical analysis applied to {len(technical_scores)} tickers." if technical_scores else ""
        summary = (
            f"Analyzed {len(stories)} stories, detected {len(valid_prices)} tradeable tickers.{ta_note} "
            f"Overall bias is {bias} with {buy_count} BUY and {sell_count} SELL signals."
        )

        return {
            "success": True,
            "analysis_summary": summary,
            "recommendations": recommendations,
            "story_details": story_details,
            "timestamp": now.isoformat(),
            "provider": "local_nlp",
            "model": "financial_lexicon_v1",
            "stories_analyzed": len(stories),
            "tickers_detected": len(valid_prices),
        }

    @staticmethod
    def _parse_age_hours(published_str: str, now: datetime) -> float:
        """Parse an RSS/ISO date string and return age in hours."""
        if not published_str:
            return 1.0  # default: treat as ~1 hour old

        # Try common RSS date formats
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",   # RFC 822
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",          # ISO 8601
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(published_str.strip(), fmt)
                # Make naive datetimes comparable
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                age = (now - dt).total_seconds() / 3600.0
                return max(0.0, age)
            except ValueError:
                continue

        return 1.0  # fallback


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = LocalTradingEngine()

    sample_stories = [
        {
            "title": "NVIDIA reports record revenue as AI chip demand soars",
            "summary": "NVIDIA's quarterly earnings beat expectations with data center revenue surging 150%. "
                       "The GPU maker sees strong demand from cloud providers for its AI accelerators.",
            "published": datetime.now().isoformat(),
        },
        {
            "title": "Oil prices drop as OPEC fails to agree on production cuts",
            "summary": "Crude oil fell 4% after OPEC members could not reach consensus on limiting output. "
                       "Exxon and Chevron shares declined in after-hours trading.",
            "published": (datetime.now() - timedelta(hours=2)).isoformat(),
        },
        {
            "title": "Federal Reserve signals potential rate cut in upcoming meeting",
            "summary": "Fed chair indicated a dovish shift, suggesting interest rates may be lowered. "
                       "Bank stocks rallied on the news while bond yields declined.",
            "published": (datetime.now() - timedelta(hours=1)).isoformat(),
        },
        {   # Tests dynamic resolution — Uber is NOT in the hardcoded alias dict
            "title": "Uber gains new robo taxi fleet as autonomous driving expands",
            "summary": "Uber announced a partnership to deploy robotaxis in major cities. "
                       "The ride-hailing giant is accelerating its autonomous vehicle strategy.",
            "published": datetime.now().isoformat(),
        },
    ]

    results = engine.analyze_stories(sample_stories)
    print(f"\n{'='*60}")
    print(f"Analysis Summary: {results['analysis_summary']}")
    print(f"Stories analyzed: {results['stories_analyzed']}")
    print(f"Tickers detected: {results['tickers_detected']}")
    print(f"{'='*60}")

    for rec in results["recommendations"]:
        print(f"\n  {rec['action']:4s} {rec['symbol']:5s} | "
              f"Confidence: {rec['confidence']*100:5.1f}% | "
              f"Allocation: {rec['target_allocation_percent']:2d}% | "
              f"Score: {rec['aggregated_score']:+.4f}")
        sent = rec.get('sentiment_score')
        tech = rec.get('technical_score')
        parts = []
        if sent is not None:
            parts.append(f"Sentiment: {sent:+.4f}")
        if tech is not None:
            parts.append(f"Technical: {tech:+.4f}")
        if parts:
            print(f"       {' | '.join(parts)}")
        print(f"       Reason: {rec['reason'][:100]}")
