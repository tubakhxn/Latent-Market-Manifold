import pandas as pd

def load_market_data(file_path):
    """
    Load historical price data from a CSV file.
    Expects columns: Date, Asset, Price (or custom columns).
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    return df
