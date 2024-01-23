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
            print("Error: 'open' not found in header.")
            return None
        
        # Get indices of open, high, low, close, volume
        indices = [h.index(s) for s in ('Open', 'High', 'Low', 'Close', 'Volume')]

        # Initialize variables
        openPrice, highPrice, lowPrice, closePrice, volume = [], [], [], [], []
        countOut, countFixed = 0, 0
        lastClose = None

        # Read in data
        for row in reader:
            # Get values
            po, ph, pl, pc, pv = list(map(float, [row[idx] for idx in indices]))

            # fix open price for current bar to match close price of last bar
            if fixOpenPrice and lastClose is not None:
                ppc = lastClose
                if abs(po - ppc) > 1e-8:
                    countFixed += 1
                    po = ppc
                    pl = min(pl, po)
                    ph = max(ph, po)
            countOut += 1
            openPrice.append(po)
            closePrice.append(pc)
            highPrice.append(ph)
            lowPrice.append(pl)
            volume.append(pv)
            lastClose = pc

    print("Read done, got %d rows, %d open prices adjusted" % (countOut, countFixed))
    return Prices(open=  np.array(openPrice,   dtype=np.float32),
                  high=  np.array(highPrice,   dtype=np.float32),
                  low=   np.array(lowPrice,    dtype=np.float32),
                  close= np.array(closePrice,  dtype=np.float32),
                  volume=np.array(volume,      dtype=np.float32))

def relativePrices(prices: Prices) -> Prices:
    """
    Convert prices to relative w.r.t to open price
    :param Prices: tuple with open, close, high, low
    :return: tuple with open, relClose, relHigh, relLow
    """

    # Check input parameters
    assert isinstance(prices, Prices)

    # Calculate relative prices
    relHigh = (prices.high - prices.open) / prices.open
    relLow = (prices.low - prices.open) / prices.open
    relClose = (prices.close - prices.open) / prices.open

    # Return tuple with relative prices
    print("Relative prices generated")
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

    # Initialize return
    result = []
    token = '*.csv'
    
    # Search for files and return
    for path in glob.glob(os.path.join(path, token)):
        result.append(path)
    return result
