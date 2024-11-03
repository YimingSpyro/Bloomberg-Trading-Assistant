import yfinance as yf
import pandas as pd
import streamlit as st

# Function to get mean Analyst Ratings and Company Name
def get_analyst_ratings(ticker):
    info = yf.Ticker(ticker).info
    return {
        'targetMeanPrice': info.get('targetMeanPrice'),
        'companyName': info.get('longName')  # Get the company name
    }

# Function to calculate potential upside based on analyst target price/current price
def calculate_upside(current_price, target_price):
    if current_price is None or target_price is None or target_price <= 0:
        return None  # Return None if current price is None or target price is invalid
    return ((target_price - current_price) / current_price) * 100

import datetime

# Function to analyze a stock and return relevant data
def analyze_stock(ticker):
    try:
        info = get_analyst_ratings(ticker)

        # Fetch up to the last 5 days of history to ensure we get Friday's close if today is a weekend
        history_data = yf.Ticker(ticker).history(period='5d')
        
        # Check if the history data is empty
        if history_data.empty:
            st.error(f"No trading data found for {ticker}.")
            current_price = None
        else:
            # Use the last valid closing price, regardless of the day
            current_price = history_data['Close'].iloc[-1]

        # Calculate potential upside if we have both current and target prices
        if info['targetMeanPrice'] is not None and current_price is not None:
            upside = calculate_upside(current_price, info['targetMeanPrice'])
        else:
            upside = None

        return {
            'Ticker': ticker,
            'Company Name': info['companyName'],
            'Current Price': current_price,
            'Target Price': info['targetMeanPrice'],
            'Potential Upside (%)': upside
        }
    except Exception as e:
        st.error(f"Error analyzing {ticker}: {str(e)}")  # Log the error for each ticker
        return {
            'Ticker': ticker,
            'Company Name': None,
            'Current Price': None,
            'Target Price': None,
            'Potential Upside (%)': None,
            'Error': str(e)
        }

# Sample usage: 
result = analyze_stock("AAPL")
print(result)

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

# Streamlit code to display the rankings
st.title("Stock Rankings Based on Potential Upside")
with st.spinner("Fetching stock data..."):  # Loading spinner
    stock_rankings = []
    for ticker in tickers:
        result = analyze_stock(ticker)
        stock_rankings.append(result)

# Convert to DataFrame and sort by Potential Upside
rankings_df = pd.DataFrame(stock_rankings)
rankings_df = rankings_df.sort_values(by='Potential Upside (%)', ascending=False)

st.table(rankings_df)
