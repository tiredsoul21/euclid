""" Data processing library """
import os
import csv
import glob
import pathlib
import collections

import numpy as np

# The container is used to store prices
Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])

# The function is used to read csv file
def read_csv(file_name: str, sep: str =',', fix_open_price: bool = False) -> Prices:
    """
    Read csv file and return a tuple with prices
    :param file_name: name of the file
    :param sep: separator (default: ',')
    :param fix_open_price: fix open price to match close price of last bar? (default: False)
    :return: tuple with prices
    """

    # Check input parameters
    assert isinstance(file_name, str)
    assert isinstance(sep, str)
    assert isinstance(fix_open_price, bool)

    # Open file and read data
    print("Reading:: ", file_name)
    with open(file_name, 'rt', encoding='utf-8') as file_descriptor:
        # Build reader and read header
        reader = csv.reader(file_descriptor, delimiter=sep)
        h = next(reader)

        # Check if 'open' is in header
        if 'Open' not in h:
            # Return an empty value set
            raise ValueError("'open' not found in header.")

        # Get indices of open, high, low, close, volume
        indices = [h.index(s) for s in ('Open', 'High', 'Low', 'Close', 'Volume')]

        # Initialize variables
        data = np.array(list(reader))

        # Extract numeric columns
        numeric_data = data[:, indices].astype(np.float32)

        open_price, high_price, low_price, close_price, volume, *_ = numeric_data.T

        # Fix open price to match close price of last bar
        if fix_open_price:
            close_price_shifted = np.roll(close_price, 1)
            close_price_shifted[0] = close_price[0]
            mask = np.abs(open_price - close_price_shifted) > 1e-8
            open_price[mask] = close_price_shifted[mask]
            low_price = np.minimum(low_price, open_price)
            high_price = np.maximum(high_price, open_price)

        # Remove rows with zero open price
        zero_price_indices = np.where(open_price == 0)[0]
        if zero_price_indices.size > 0:
            print(f"File {file_name} has zero open price til {zero_price_indices[-1]}")
            zero_price_index = zero_price_indices[-1]
            open_price = open_price[zero_price_index + 1:]
            close_price = close_price[zero_price_index + 1:]
            high_price = high_price[zero_price_index + 1:]
            low_price = low_price[zero_price_index + 1:]
            volume = volume[zero_price_index + 1:]

    return Prices(open=open_price, high=high_price, low=low_price, close=close_price, volume=volume)

def relative_prices(prices: Prices) -> Prices:
    """
    Convert prices to relative w.r.t to open price
    :param prices: tuple with open, close, high, low
    :return: tuple with open, rel_close, rel_high, rel_low
    """

    # Check input parameters
    assert isinstance(prices, Prices)

    # Calculate relative prices using vectorized operations
    rel_high = (prices.high - prices.open) / prices.open
    rel_low = (prices.low - prices.open) / prices.open
    rel_close = (prices.close - prices.open) / prices.open

    # Return tuple with relative prices
    return Prices(open=prices.open,
                  high=rel_high,
                  low=rel_low,
                  close=rel_close,
                  volume=prices.volume)

def find_files(path: pathlib.Path) -> list:
    """
    Find files in directory
    :param directory: directory to search in
    :return: list of files
    """

    # Check input parameters
    assert isinstance(path, pathlib.Path)

    # Search for files and return
    return [path for path in glob.glob(os.path.join(path, '*.csv'))]

def load_relative(csv_file: str, sep: str =',', fix_open_price: bool = False) -> Prices:
    """
    Load relative prices from csv file
    :param csv_file: csv file to load
    :return: tuple with relative prices
    """
    return relative_prices(read_csv(csv_file, sep, fix_open_price))
