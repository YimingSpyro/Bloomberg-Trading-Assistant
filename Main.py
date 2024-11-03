import yfinance as yf
import pandas as pd
import streamlit as st

# Function to get mean Analyst Ratings and Company Name
def get_analyst_ratings(ticker):
    info = yf.Ticker(ticker).info
    return {
        'targetMeanPrice': info.get('targetMeanPrice'),
        'earningsDate': info.get('earningsDate'),
        'companyName': info.get('longName')  # Get the company name
    }

# Function to calculate potential upside based on analyst target price/current price
def calculate_upside(current_price, target_price):
    if target_price <= 0:
        raise ValueError("Target price must be positive")
    return ((target_price - current_price) / current_price) * 100

# Function to analyze a stock and return relevant data
def analyze_stock(ticker):
    try:
        info = get_analyst_ratings(ticker)
        current_price_data = yf.Ticker(ticker).history(period='1d')

        # Check if the history data is empty
        if current_price_data.empty:
            current_price = None
        else:
            current_price = current_price_data['Close'].iloc[-1]

        if info['targetMeanPrice'] is not None:
            try:
                upside = calculate_upside(current_price, info['targetMeanPrice'])
            except ValueError:
                upside = None
        else:
            upside = None

        return {
            'Ticker': ticker,
            'Company Name': info['companyName'],  # Add company name to return
            'Current Price': current_price,
            'Target Price': info['targetMeanPrice'],
            'Potential Upside (%)': upside,
            'Earnings Date': info['earningsDate']
        }
    except Exception as e:
        return {
            'Ticker': ticker,
            'Company Name': None,  # Set company name to None on error
            'Current Price': None,
            'Target Price': None,
            'Potential Upside (%)': None,
            'Earnings Date': None,
            'Error': str(e)
        }

# Define the tickers for analysis
tickers = [
    "SAP",     # SAP (SAP)
    "PNGAY",   # Ping An Insurance (PNGAY)
    "LOGI",    # Logitech (LOGI)
    "F",       # Ford
    "PLTR",    # Palantir Technologies (PLTR)
    "VRTX",    # Vertex Pharmaceuticals (VRTX)
    "RTX",     # Raytheon (RTX)
    "VZ",      # Verizon (VZ)
    "MCO",     # Moody’s (MCO)
    "GM",      # General Motors (GM)
    "LMT",     # Lockheed (LMT)
    "PYPL",    # PayPal Holdings (PYPL)
    "PFE",     # Pfizer (PFE)
    "AMD",     # Advanced Micro Devices (AMD)
    "GOOG",    # Alphabet Inc C (GOOG)
    "GOOGL",   # Alphabet Inc A (GOOGL)
    "SNAP",    # Snap Inc A (SNAP)
    "V",       # Visa Inc Class A (V)
    "SPOT",    # Spotify Technology SA (SPOT)
    "HD",      # The Home Depot (HD)
    "OXY",     # Occidental Petroleum (OXY)
    "BA",      # Boeing (BA)
    "KO",      # Coca Cola (KO)
    "HLT",     # Hilton (HLT)
    "T",       # AT&T (T)
    "DB",      # Deutsche Bank (DB)
    "TSLA",    # Tesla (TSLA)
    "IBM",     # IBM (IBM)
    "MORN",    # Morningstar (MORN)
    "KPELY",   # Keppel (KPELY)
    "ETSY",    # Etsy (ETSY)
    "FVRR",    # Fiverr International (FVRR)
    "KHC",     # The Kraft Heinz Co (KHC)
    "BKNG",    # Booking Holdings (BKNG)
    "META",    # Meta Platforms (META)
    "MSFT",    # Microsoft Corp (MSFT)
    "SBUX",    # Starbucks Corp (SBUX)
    "ABNB",    # Airbnb (ABNB)
    "QCOM",    # Qualcomm Inc (QCOM)
    "SMCI",    # Super Micro Computer (SMCI)
    "RBLX",    # Roblox (RBLX)
    "AMC",     # AMC Entertainment (AMC)
    "MARA",    # Marathon Digital (MARA)
    "RIOT",    # Riot Platforms (RIOT)
    "NDAQ",    # Nasdaq (NDAQ)
    "UPS",     # UPS (UPS)
    "BCS",     # Barclays (BCS)
    "AMZN",    # Amazon.com (AMZN)
    "AAPL",    # Apple Inc (AAPL)
    "INTC",    # Intel Corp (INTC)
    "GOLD",    # Barrick Gold (GOLD)
    "MRNA",    # Moderna (MRNA)
    "SHOP",    # Shopify (SHOP)
    "NET",     # Cloudflare (NET)
    "COIN",    # Coinbase Global (COIN)
    "LCID",    # Lucid Group (LCID)
    "PINS",    # Pinterest (PINS)
    "RIVN",    # Rivian Automotive (RIVN)
    "DIS",     # The Walt Disney Co (DIS)
    "ILMN",    # Illumina (ILMN)
    "TTD",     # The Trade Desk (TTD)
    "U",       # Unity Software (U)
    "CVX",     # Chevron (CVX)
    "XOM"      # Exxon Mobil (XOM)
]

# Analyze all tickers and store results in a DataFrame
stock_rankings = []
for ticker in tickers:
    result = analyze_stock(ticker)
    stock_rankings.append(result)

# Convert to DataFrame and sort by Potential Upside
rankings_df = pd.DataFrame(stock_rankings)
rankings_df = rankings_df.sort_values(by='Potential Upside (%)', ascending=False)

# Streamlit code to display the rankings
st.title("Stock Rankings Based on Potential Upside")
st.table(rankings_df)
