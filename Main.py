import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF
import os

# List of stocks to analyze
tickers = [
    'SAP', 'PNGAY', 'LOGI', 'F', 'PLTR', 'VRTX', 'RTX', 'VZ', 'MCO', 
    'GM', 'LMT', 'PYPL', 'PFE', 'AMD', 'GOOG', 'GOOGL', 'SNAP', 'V', 
    'SPOT', 'HD', 'OXY', 'BA', 'KO', 'HLT', 'T', 'DB', 'TSLA', 'IBM', 
    'MORN', 'KPELY', 'ETSY', 'FVRR', 'KHC', 'BKNG', 'META', 'MSFT', 
    'SBUX', 'ABNB', 'QCOM', 'SMCI', 'RBLX', 'AMC', 'MARA', 'RIOT', 
    'NDAQ', 'UPS', 'NOC', 'BCS', 'AMZN', 'UBER', 'AMZN', 'AAPL', 'INTC', 
    'GOLD', 'MRNA', 'SHOP', 'NET', 'COIN', 'LCID', 'PINS', 'RIVN', 'DIS', 
    'ILMN', 'TTD', 'U', 'CSUA', 'CVX', 'XOM'
]


# Function to calculate technical indicators
def calculate_indicators(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ma = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ma
    signal = macd.ewm(span=9, adjust=False).mean()

    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    sma = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)

    return macd, signal, rsi, upper_band, lower_band

# Function to identify Buy and Sell Signals
def define_signals(data):
    data.loc[:, 'Buy_Signal'] = ((data['MACD'] < data['Signal']) & (data['MACD'] < 0) & (data['RSI'] < 30) & (data['Close'] <= data['Lower_Band'])).rolling(window=5).sum() >= 1
    data.loc[:, 'Sell_Signal'] = ((data['MACD'] > data['Signal']) & (data['MACD'] > 0) & (data['RSI'] > 70) & (data['Close'] >= data['Upper_Band'])).rolling(window=5).sum() >= 1
    return data

# Function to calculate average sell signals
def average_sell_signals(data):
    sell_count = 0
    buy_count = 0
    buy_state = 0
    total_sell_signals_between_buys = 0

    for i in range(len(data)):
        if data['Buy_Signal'].iloc[i]:
            buy_state += 1
            if buy_state >= 1:
                total_sell_signals_between_buys += sell_count
                sell_count = 0  
            
        elif data['Sell_Signal'].iloc[i]:
            if buy_state >= 1:
                buy_state = 0
                buy_count += 1
            sell_count += 1

    average = total_sell_signals_between_buys / buy_count if buy_count > 0 else 0
    return average

# Function for Trading algorithm
def trading_strategy(data, avg_sell_signals, initial_amount=10000):
    balance = initial_amount
    stock_quantity = 0
    profits = []
    in_trade = False
    buy_price = None
    sell_signal_count = 0

    sell_threshold = avg_sell_signals / 2

    for i in range(len(data)):
        if data['Buy_Signal'].iloc[i] and not in_trade:
            stock_quantity = balance / data['Close'].iloc[i]
            balance = 0
            buy_price = data['Close'].iloc[i]
            in_trade = True
            sell_signal_count = 0 
            
        elif data['Sell_Signal'].iloc[i] and in_trade:
            sell_signal_count += 1
            
            if sell_signal_count >= sell_threshold:
                balance = stock_quantity * data['Close'].iloc[i]
                stock_quantity = 0
                sell_price = data['Close'].iloc[i]
                profit_percent = ((sell_price - buy_price) / buy_price) * 100
                profits.append(profit_percent)
                in_trade = False
                sell_signal_count = 0

    if in_trade:
        balance = stock_quantity * data['Close'].iloc[-1]
    
    final_amount = balance
    return profits, final_amount, in_trade  # Return in_trade status

# Function to calculate/record trading strategy performance
def calculate_performance(profits, initial_amount, final_amount, period_years):
    total_trades = len(profits)
    avg_return_per_trade = sum(profits) / total_trades if total_trades > 0 else 0
    total_return = ((final_amount - initial_amount) / initial_amount) * 100
    avg_annual_return = ((final_amount / initial_amount) ** (1 / period_years) - 1) * 100 if period_years > 0 else 0
    return total_trades, avg_return_per_trade, total_return, avg_annual_return

# Function to get mean Analyst Ratings
def get_analyst_ratings(ticker):
    info = yf.Ticker(ticker).info
    return {
        'targetMeanPrice': info.get('targetMeanPrice')
    }

# Function to calculate potential upside based on analyst target price/current price
def calculate_upside(current_price, target_price):
    if target_price <= 0:
        raise ValueError("Target price must be positive")
    return ((target_price - current_price) / current_price) * 100

# Function to extract 'Date' and 'Close' columns, and fill missing 'Close' values with the previous day's value
def clean(data):
    data = data[['Close']]
    data.reset_index(inplace=True)  # Reset index to have 'Date' as a column
    data['Close'].fillna(method='ffill', inplace=True)
    return data

# Display trading algorithm performance
def analyze_stock(ticker):
    data = clean(yf.download(ticker, period='10y'))

    train_data = data[:-252]
    test_data = data[-252:]

    train_data['MACD'], train_data['Signal'], train_data['RSI'], train_data['Upper_Band'], train_data['Lower_Band'] = calculate_indicators(train_data)
    test_data['MACD'], test_data['Signal'], test_data['RSI'], test_data['Upper_Band'], test_data['Lower_Band'] = calculate_indicators(test_data)

    train_data = define_signals(train_data)
    test_data = define_signals(test_data)

    avg_sell_signals_train = average_sell_signals(train_data)
    avg_sell_signals_test = average_sell_signals(test_data)

    test_profits, test_final_balance, in_trade = trading_strategy(test_data, avg_sell_signals_test)

    test_years = len(test_data) / 252
    test_total_trades, test_avg_return_per_trade, test_total_return, test_avg_annual_return = calculate_performance(
        test_profits, 10000, test_final_balance, test_years)

    info = get_analyst_ratings(ticker)
    target_price = info.get('targetMeanPrice')
    current_price = test_data['Close'].iloc[-1]
        
    if target_price is not None:
        try:
            upside = calculate_upside(current_price, target_price)
        except ValueError:
            upside = None
    else:
        upside = None

    return {
        'Ticker': ticker,
        'Current Price ($)': round(current_price, 2),
        'Target Price ($)': round(target_price, 2) if target_price is not None else None,
        'Potential Upside (%)': round(upside, 2) if upside is not None else None,
        'Trades Closed': test_total_trades,
        'Average Return per Trade (%)': round(test_avg_return_per_trade, 2),
        'Total Return (%)': round(test_total_return, 2),
        'In Trade': in_trade,  # Indicate if still in trade
        'Train Data': train_data,
        'Test Data': test_data
    }

def plot_or_save_stock_data(ticker, test_data, action='show'):
    # Ensure the DataFrame is a copy to avoid view/copy issues
    test_data = test_data.copy()

    # Check if the necessary columns exist
    required_columns = ['Date', 'Close', 'Upper_Band', 'Lower_Band', 'MACD', 'Signal', 'RSI', 'Buy_Signal', 'Sell_Signal']
    for column in required_columns:
        if column not in test_data.columns:
            raise ValueError(f"The '{column}' column is missing from the test data.")

    # Convert 'Date' to datetime if not already
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    
    # Set 'Date' as the index
    test_data.set_index('Date', inplace=True)

    # Prepare data for plotting
    upper_band = test_data['Upper_Band']
    lower_band = test_data['Lower_Band']
    sma_20 = test_data['Close'].rolling(window=20).mean()
    macd = test_data['MACD']
    signal = test_data['Signal']
    rsi = test_data['RSI']
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(14, 10), sharex=True)

    # Plot closing price, Bollinger Bands, Moving Averages, and Buy/Sell signals
    ax1.plot(test_data['Close'], label='Close Price', color='black')
    ax1.plot(upper_band, label='Upper Bollinger Band', color='red')
    ax1.plot(lower_band, label='Lower Bollinger Band', color='blue')
    ax1.plot(sma_20, label='20-day SMA', color='orange')
    ax1.fill_between(test_data.index, lower_band, upper_band, color='gray', alpha=0.3)
    ax1.scatter(test_data.index[test_data['Buy_Signal']], 
                test_data['Close'][test_data['Buy_Signal']], 
                label='Buy Signal', marker='^', color='green', alpha=1)
    ax1.scatter(test_data.index[test_data['Sell_Signal']], 
                test_data['Close'][test_data['Sell_Signal']], 
                label='Sell Signal', marker='v', color='red', alpha=1)
    ax1.set_title(f'{ticker} Stock Price, Bollinger Bands, Moving Averages, Buy & Sell Signals')
    ax1.legend(loc='upper left')  # Move legend to the top left

    # Plot MACD and Signal Line
    ax2.plot(macd, label='MACD', color='green')
    ax2.plot(signal, label='Signal Line', color='red')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('MACD and Signal Line')
    ax2.legend(loc='upper left')  # Move legend to the top left

    # Plot RSI
    ax3.plot(rsi, label='RSI', color='purple')
    ax3.axhline(30, linestyle='--', alpha=0.5, color='red')
    ax3.axhline(70, linestyle='--', alpha=0.5, color='red')
    ax3.set_title('RSI')
    ax3.legend(loc='upper left')  # Move legend to the top left

    # Use Streamlit to display plots
    if action == 'save':
        # Ensure the output directory exists
        output_dir = 'saved_plots'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_path = os.path.join(output_dir, f'{ticker}_signals.png')
        plt.savefig(image_path)
        plt.close()
        return image_path
    else:
        # Display the plot in Streamlit
        st.pyplot(fig)
        plt.close()  # Close the figure after displaying to avoid overlapping plots


# Main Streamlit Program
def main():
    st.title("Stock Analysis and Trading Strategy")
    st.write("1. Select the stocks you want to analyse.")
    st.write("2. Review our analysis and trading strategy performance.")
    st.write("3. Save as pdf for future reference.")
    
    # Create a checkbox for each ticker to analyze
    selected_tickers = st.multiselect("Select Tickers to Analyze", tickers)
    
    if selected_tickers:
        results = []
        top_stock_images = []  # Store paths for top stock images
        for ticker in selected_tickers:
            result = analyze_stock(ticker)
            results.append(result)

        # Stock Ranking Section
        results_df = pd.DataFrame(results)
        stock_ranking_df = results_df.sort_values(by='Potential Upside (%)', ascending=False).copy()
        stock_ranking_df['Rank'] = results_df['Potential Upside (%)'].rank(ascending=False) 
        trade_bot_df = results_df.sort_values(by='Total Return (%)', ascending=False).copy()
        trade_bot_df['Rank'] = results_df['Total Return (%)'].rank(ascending=False) 
        

        st.write("### Stock Ranking:")
        st.write("*Current Price refers to the last closing price of training dataset.")
        st.write("*Target Price refers to the mean analyst price target.")
        st.dataframe(stock_ranking_df[['Rank', 'Ticker', 'Current Price ($)', 'Target Price ($)', 'Potential Upside (%)']])

        # Trade Bot Performance Section
        st.write("### Trade Bot Performance:")
        st.write("*Trading performance based on 1 year of test data.")
        performance_df = trade_bot_df[['Rank','Ticker', 'Trades Closed', 'Average Return per Trade (%)', 'Total Return (%)', 'In Trade']].copy()
        performance_df.rename(columns={'In Trade': 'Status'}, inplace=True)
        performance_df['Status'] = performance_df['Status'].apply(lambda x: "In Trade" if x else "Closed")  # Set Status
        st.dataframe(performance_df)

        # Graph Section for Top 3 Stocks
        st.write("### Top 3 stocks:")
        top_stocks = trade_bot_df.head()
        for index, row in top_stocks.iterrows():
            st.write(f"{row['Ticker']} Stock Data")
            plot_or_save_stock_data(row['Ticker'], row['Test Data'], action='plot')

# Run the Streamlit app
if __name__ == "__main__":
    main()
