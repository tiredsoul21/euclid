import os
import csv
import glob
import pathlib
import numpy as np
import collections

# The container is used to store prices
Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])

# The function is used to read csv file
def readCSV(fileName: str, sep: str =',', fixOpenPrice: bool = False) -> Prices:
    """
    Read csv file and return a tuple with prices
    :param fileName: name of the file
    :param sep: separator (default: ',')
    :param fixOpenPrice: fix open price to match close price of last bar? (default: False)
    :return: tuple with prices
    """

    # Check input parameters
    assert isinstance(fileName, str)
    assert isinstance(sep, str)
    assert isinstance(fixOpenPrice, bool)

    # Open file and read data
    print("Reading:: ", fileName)
    with open(fileName, 'rt', encoding='utf-8') as fileDescriptor:
        # Build reader and read header
        reader = csv.reader(fileDescriptor, delimiter=sep)
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

        openPrice, highPrice, lowPrice, closePrice, volume, *_ = numeric_data.T

        # Fix open price to match close price of last bar
        if fixOpenPrice:
            closePriceShifted = np.roll(closePrice, 1)
            closePriceShifted[0] = closePrice[0]
            mask = np.abs(openPrice - closePriceShifted) > 1e-8
            openPrice[mask] = closePriceShifted[mask]
            lowPrice = np.minimum(lowPrice, openPrice)
            highPrice = np.maximum(highPrice, openPrice)

        # Remove rows with zero open price
        zeroPriceIndices = np.where(openPrice == 0)[0]
        if zeroPriceIndices.size > 0:
            print(f"File {fileName} has zero open price til {zeroPriceIndices[-1]}")
            lastZeroIdx = zeroPriceIndices[-1]
            openPrice = openPrice[lastZeroIdx + 1:]
            closePrice = closePrice[lastZeroIdx + 1:]
            highPrice = highPrice[lastZeroIdx + 1:]
            lowPrice = lowPrice[lastZeroIdx + 1:]
            volume = volume[lastZeroIdx + 1:]

    return Prices(open=openPrice, high=highPrice, low=lowPrice, close=closePrice, volume=volume)

def relativePrices(prices: Prices) -> Prices:
    """
    Convert prices to relative w.r.t to open price
    :param Prices: tuple with open, close, high, low
    :return: tuple with open, relClose, relHigh, relLow
    """

    # Check input parameters
    assert isinstance(prices, Prices)

    # Calculate relative prices using vectorized operations
    relHigh = (prices.high - prices.open) / prices.open
    relLow = (prices.low - prices.open) / prices.open
    relClose = (prices.close - prices.open) / prices.open

    # Return tuple with relative prices
    return Prices(open=prices.open, 
                  high=relHigh,
                  low=relLow, 
                  close=relClose, 
                  volume=prices.volume)

def findFiles(path: pathlib.Path) -> list:
    """
    Find files in directory
    :param directory: directory to search in
    :return: list of files
    """

    # Check input parameters
    assert isinstance(path, pathlib.Path)

    # Search for files and return
    return [path for path in glob.glob(os.path.join(path, '*.csv'))]

def loadRelative(csvFile: str, sep: str =',', fixOpenPrice: bool = False) -> Prices:
    """
    Load relative prices from csv file
    :param csvFile: csv file to load
    :return: tuple with relative prices
    """
    return relativePrices(readCSV(csvFile, sep, fixOpenPrice))