import os
import shutil
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Global Variables and Configurations
# ---------------------------------------------------------------------
stock_symbol = ['VZ']

API_KEY = 'XKV2M77TLLDJHXQK'



# ---------------------------------------------------------------------
# Data Fetching Functions
# ---------------------------------------------------------------------
def fetch_historical_stock_data(symbol):
    """
    Fetches historical stock prices using yfinance and saves them to CSV.
    """
    try:
        stock = yf.Ticker(symbol)
        today = datetime.today().strftime('%Y-%m-%d')
        df = stock.history(start="2015-01-01", end=today)

        if df.empty:
            print(f"No historical data found for {symbol}")
            return
        
        print(f"Columns in historical data: {df.columns}")
        df.reset_index(inplace=True)

        # Ensure 'Adj Close' exists, or rename 'Close' if needed
        if 'Adj Close' not in df.columns:
            if 'Close' in df.columns:
                df.rename(columns={'Close': 'Adj Close'}, inplace=True)
            else:
                print(f"'Adj Close' or 'Close' column not found for {symbol}")
                return
        
        # Save inside the "AAPL" folder
        output_path = os.path.join(output_folder, f'{symbol}_historical_data.csv')
        df[['Date', 'Adj Close', 'Volume']].to_csv(output_path, index=False)
        print(f"Historical stock data for {symbol} saved to {output_path}.")

    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")


def fetch_balance_sheet_data(symbol, api_key):
    """
    Fetches balance sheet data from AlphaVantage (quarterly) and saves to CSV.
    """
    try:
        url = (f'https://www.alphavantage.co/query?function=BALANCE_SHEET'
               f'&symbol={symbol}&apikey={api_key}')
        response = requests.get(url)
        data = response.json()

        if 'quarterlyReports' in data:
            df = pd.DataFrame(data['quarterlyReports'])
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df[df['fiscalDateEnding'] >= '2015-01-01']

            output_path = os.path.join(output_folder, f'{symbol}_balance_sheet_quarterly.csv')
            df.to_csv(output_path, index=False)
            print(f"Balance sheet data for {symbol} saved to {output_path}.")
        else:
            print(f"Error fetching balance sheet data for {symbol}: {data}")

    except Exception as e:
        print(f"Exception occurred while fetching balance sheet data for {symbol}: {e}")


def fetch_cash_flow_data(symbol, api_key):
    """
    Fetches cash flow data from AlphaVantage (quarterly) and saves to CSV.
    """
    try:
        url = (f'https://www.alphavantage.co/query?function=CASH_FLOW'
               f'&symbol={symbol}&apikey={api_key}')
        response = requests.get(url)
        data = response.json()

        if 'quarterlyReports' in data:
            df = pd.DataFrame(data['quarterlyReports'])
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df[df['fiscalDateEnding'] >= '2015-01-01']

            output_path = os.path.join(output_folder, f'{symbol}_cash_flow_quarterly.csv')
            df.to_csv(output_path, index=False)
            print(f"Cash flow data for {symbol} saved to {output_path}.")
        else:
            print(f"Error fetching cash flow data for {symbol}: {data}")

    except Exception as e:
        print(f"Exception occurred while fetching cash flow data for {symbol}: {e}")


def fetch_income_statement_data(symbol, api_key):
    """
    Fetches income statement data from AlphaVantage (quarterly) and saves to CSV.
    """
    try:
        url = (f'https://www.alphavantage.co/query?function=INCOME_STATEMENT'
               f'&symbol={symbol}&apikey={api_key}')
        response = requests.get(url)
        data = response.json()

        if 'quarterlyReports' in data:
            df = pd.DataFrame(data['quarterlyReports'])
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df[df['fiscalDateEnding'] >= '2015-01-01']

            output_path = os.path.join(output_folder, f'{symbol}_income_statement_quarterly.csv')
            df.to_csv(output_path, index=False)
            print(f"Income statement data for {symbol} saved to {output_path}.")
        else:
            print(f"Error fetching income statement data for {symbol}: {data}")

    except Exception as e:
        print(f"Exception occurred while fetching income statement data for {symbol}: {e}")


def fetch_all_data(symbol, api_key):
    """
    Fetches all necessary data for a given stock symbol.
    """
    fetch_historical_stock_data(symbol)
    fetch_balance_sheet_data(symbol, api_key)
    fetch_cash_flow_data(symbol, api_key)
    fetch_income_statement_data(symbol, api_key)



# ---------------------------------------------------------------------
# Data Processing and Merging
# ---------------------------------------------------------------------
def process_and_merge_data(symbol):
    """
    Processes fetched data, calculates financial metrics, merges datasets,
    and adds rolling Beta using SP500 data.
    """
    # 1. Load historical data
    hist_path = os.path.join(output_folder, f'{symbol}_historical_data.csv')
    historical_data = pd.read_csv(hist_path)
    historical_data['Date'] = pd.to_datetime(historical_data['Date'], utc=True)

    final_data = historical_data[['Date', 'Adj Close', 'Volume']]
    final_data_path = os.path.join(output_folder, f'formatted_{symbol}_data.csv')
    final_data.to_csv(final_data_path, index=False)
    print(f"Formatted data saved to '{final_data_path}'")

    # 2. Re-load formatted data + balance sheet
    historical_data = pd.read_csv(final_data_path)
    historical_data['Date'] = pd.to_datetime(historical_data['Date'], utc=True)

    bs_path = os.path.join(output_folder, f'{symbol}_balance_sheet_quarterly.csv')
    balance_sheet_data = pd.read_csv(bs_path)
    balance_sheet_data['fiscalDateEnding'] = pd.to_datetime(balance_sheet_data['fiscalDateEnding'], utc=True)

    # 3. DE Ratio
    balance_sheet_data['DE Ratio'] = balance_sheet_data['totalLiabilities'] / balance_sheet_data['totalShareholderEquity']
    balance_sheet_data.set_index('fiscalDateEnding', inplace=True)
    historical_data.set_index('Date', inplace=True)

    balance_sheet_data = balance_sheet_data.reindex(historical_data.index, method='ffill')
    historical_data['DE Ratio'] = balance_sheet_data['DE Ratio']
    historical_data.reset_index(inplace=True)
    historical_data.to_csv(final_data_path, index=False)
    print(f"DE Ratio added to '{final_data_path}'")

    # 4. Add ROE
    balance_sheet = pd.read_csv(bs_path)
    is_path = os.path.join(output_folder, f'{symbol}_income_statement_quarterly.csv')
    income_statement = pd.read_csv(is_path)
    existing_data = pd.read_csv(final_data_path)

    balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
    income_statement['fiscalDateEnding'] = pd.to_datetime(income_statement['fiscalDateEnding'])

    financial_data = pd.merge(
        balance_sheet[['fiscalDateEnding', 'totalShareholderEquity']],
        income_statement[['fiscalDateEnding', 'netIncome']],
        on='fiscalDateEnding', how='inner'
    )
    financial_data['ROE'] = financial_data['netIncome'] / financial_data['totalShareholderEquity']
    financial_data['Year'] = financial_data['fiscalDateEnding'].dt.year
    financial_data['Quarter'] = financial_data['fiscalDateEnding'].dt.quarter

    existing_data['Year'] = pd.to_datetime(existing_data['Date']).dt.year
    existing_data['Quarter'] = pd.to_datetime(existing_data['Date']).dt.quarter

    combined_data = pd.merge(
        existing_data, financial_data[['Year', 'Quarter', 'ROE']],
        on=['Year', 'Quarter'], how='left'
    )
    combined_data['ROE'] = combined_data['ROE'].ffill()
    combined_data.drop(columns=['Year', 'Quarter'], inplace=True)
    combined_data.to_csv(final_data_path, index=False)
    print(f"ROE added to '{final_data_path}'")

    # 5. Add Price/Book
    balance_sheet = pd.read_csv(bs_path)
    existing_data = pd.read_csv(final_data_path)
    balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
    financial_data = balance_sheet[['fiscalDateEnding', 'totalShareholderEquity', 'commonStockSharesOutstanding']]

    existing_data['Date'] = pd.to_datetime(existing_data['Date'])
    existing_data['Year'] = existing_data['Date'].dt.year
    existing_data['Quarter'] = existing_data['Date'].dt.quarter

    financial_data['Price/Book'] = (
        existing_data['Adj Close'] / 
        (financial_data['totalShareholderEquity'] / financial_data['commonStockSharesOutstanding'])
    )
    financial_data['Year'] = financial_data['fiscalDateEnding'].dt.year
    financial_data['Quarter'] = financial_data['fiscalDateEnding'].dt.quarter

    combined_data = pd.merge(
        existing_data, financial_data[['Year', 'Quarter', 'Price/Book']],
        on=['Year', 'Quarter'], how='left'
    )
    combined_data['Price/Book'] = combined_data['Price/Book'].ffill()
    combined_data.drop(columns=['Year', 'Quarter'], inplace=True)
    combined_data.to_csv(final_data_path, index=False)
    print(f"Price/Book added to '{final_data_path}'")

    # 6. Add Profit Margin
    combined_data = pd.read_csv(final_data_path, parse_dates=['Date'])
    income_statement = pd.read_csv(is_path)
    income_statement['fiscalDateEnding'] = pd.to_datetime(income_statement['fiscalDateEnding'])
    financial_data = income_statement[['fiscalDateEnding', 'totalRevenue', 'netIncome']]
    financial_data['Profit Margin'] = financial_data['netIncome'] / financial_data['totalRevenue']
    financial_data['Year'] = financial_data['fiscalDateEnding'].dt.year
    financial_data['Quarter'] = financial_data['fiscalDateEnding'].dt.quarter

    combined_data['Year'] = combined_data['Date'].dt.year
    combined_data['Quarter'] = combined_data['Date'].dt.quarter

    combined_data = pd.merge(
        combined_data, financial_data[['Year', 'Quarter', 'Profit Margin']],
        on=['Year', 'Quarter'], how='left'
    )
    combined_data['Profit Margin'] = combined_data['Profit Margin'].ffill()
    combined_data.drop(columns=['Year', 'Quarter'], inplace=True)
    combined_data.to_csv(final_data_path, index=False)
    print(f"Profit Margin added to '{final_data_path}'")

    # 7. Add Diluted EPS
    balance_sheet = pd.read_csv(bs_path)
    income_statement = pd.read_csv(is_path)
    balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
    income_statement['fiscalDateEnding'] = pd.to_datetime(income_statement['fiscalDateEnding'])

    financial_data = pd.merge(balance_sheet, income_statement, on='fiscalDateEnding', how='left')
    financial_data['dilutedEPS'] = financial_data['netIncome'] / financial_data['commonStockSharesOutstanding']
    financial_data = financial_data[['fiscalDateEnding', 'dilutedEPS']]

    combined_data = pd.read_csv(final_data_path)
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])
    combined_data['Year'] = combined_data['Date'].dt.year
    combined_data['Quarter'] = combined_data['Date'].dt.to_period('Q')

    financial_data['Year'] = financial_data['fiscalDateEnding'].dt.year
    financial_data['Quarter'] = financial_data['fiscalDateEnding'].dt.to_period('Q')

    combined_data = pd.merge(
        combined_data, financial_data, on=['Year', 'Quarter'], how='left'
    )
    combined_data['dilutedEPS'] = combined_data['dilutedEPS'].ffill()
    combined_data.to_csv(final_data_path, index=False)
    print(f"Diluted EPS added to '{final_data_path}'")

    # 8. Add Rolling Beta with SP500
    stock_data = pd.read_csv(final_data_path, parse_dates=['Date'])

    # If you've copied SP500_data.csv into the "AAPL" folder, read from there:
    # sp500_file = os.path.join(output_folder, 'SP500_data.csv')
    #
    # Otherwise, read it from the top-level directory:
    sp500_file = 'SP500_data.csv'  # or another path if needed

    sp500_data = pd.read_csv(sp500_file, parse_dates=['Date'])
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.tz_localize(None)
    sp500_data['Date'] = pd.to_datetime(sp500_data['Date'], utc=True).dt.tz_localize(None)

    merged_data = pd.merge(stock_data, sp500_data, on='Date', suffixes=('_stock', '_sp500'))

    # Calculate daily returns
    merged_data['Returns_stock'] = merged_data['Adj Close_stock'].pct_change()
    merged_data['Returns_sp500'] = merged_data['Adj Close_sp500'].pct_change()
    merged_data.dropna(subset=['Returns_stock', 'Returns_sp500'], inplace=True)

    # Define rolling window size (~1 year = 252 trading days)
    window_size = 252
    rolling_cov = merged_data['Returns_stock'].rolling(window=window_size).cov(merged_data['Returns_sp500'])
    rolling_var_sp500 = merged_data['Returns_sp500'].rolling(window=window_size).var()

    merged_data['Beta'] = rolling_cov / rolling_var_sp500
    merged_data['Beta'] = merged_data['Beta'].ffill()

    beta_file = os.path.join(output_folder, f'formatted_{symbol}_data_with_rolling_beta.csv')
    merged_data.to_csv(beta_file, index=False)
    print(f"Updated dataset with rolling Beta saved to '{beta_file}'")

# ---------------------------------------------------------------------
# Data Cleaning Functions
# ---------------------------------------------------------------------
def clean_stock_data(stock_symbol):
    """
    Cleans the stock data by renaming columns and removing unnecessary columns.
    """
    input_file = os.path.join(output_folder, f'formatted_{stock_symbol}_data_with_rolling_beta.csv')
    output_file = os.path.join(output_folder, f'cleaned_{stock_symbol}_data.csv')

    data = pd.read_csv(input_file)

    column_mapping = {
        'Date': 'Date',
        'Adj Close_stock': 'adjusted_close_price',
        'Volume_stock': 'trading_volume',
        'DE Ratio': 'debt_to_equity_ratio',
        'ROE': 'return_on_equity',
        'Price/Book': 'price_to_book_ratio',
        'Profit Margin': 'profit_margin',
        'dilutedEPS': 'diluted_earnings_per_share',
        'Beta': 'company_beta'
    }

    data = data.rename(columns=column_mapping)

    columns_to_keep = [
        'Date',
        'adjusted_close_price',
        'trading_volume',
        'debt_to_equity_ratio',
        'return_on_equity',
        'price_to_book_ratio',
        'profit_margin',
        'diluted_earnings_per_share',
        'company_beta'
    ]
    data = data[columns_to_keep]

    data.to_csv(output_file, index=False)
    print(f"Column names and unnecessary columns removed. Data saved to '{output_file}'")


def replace_empty_values_with_mean(stock_symbol):
    """
    Replaces empty values in the stock data with the average value of the column.
    """
    input_file = os.path.join(output_folder, f'cleaned_{stock_symbol}_data.csv')
    data = pd.read_csv(input_file)

    for column in data.columns:
        if data[column].isnull().any() or (data[column] == '').any():
            col_mean = data[column].replace('', pd.NA).astype(float).mean()
            data[column].replace('', col_mean, inplace=True)
            data[column].fillna(col_mean, inplace=True)

    data.to_csv(input_file, index=False)
    print(f"Empty values replaced with column averages. Data updated in '{input_file}'")

# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":

    for ticker in stock_symbol:
        try:
            output_folder = ticker
            os.makedirs(output_folder, exist_ok=True)

            # 1. Fetch raw data
            fetch_all_data(ticker, API_KEY)

            # 2. Process and merge data (adds financial ratios, Beta, etc.)
            process_and_merge_data(ticker)

            # 3. Clean and finalize data
            clean_stock_data(ticker)
            replace_empty_values_with_mean(ticker)
        
        except Exception as e:
            print(f"An error occurred with ticker {ticker}: {e}")
            continue

    
