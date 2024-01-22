import os
import csv
import glob
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
        if 'open' not in h:
            # Return an empty value set
            print("Error: 'open' not found in header.")
            return Prices(open=np.array([],   dtype=np.float32),
                          high=np.array([],   dtype=np.float32),
                          low=np.array([],    dtype=np.float32),
                          close=np.array([],  dtype=np.float32),
                          volume=np.array([], dtype=np.float32))
        
        # Get indices of open, high, low, close, volume
        indices = np.array([h.index(s) for s in ('open', 'high', 'low', 'close', 'volume')])

        # Initialize variables
        data = np.empty((5, 0), dtype=np.float32)
        countOut, countFixed = 0, 0
        lastClose = None

        # Read in data
        for row in reader:
            # Get values
            vals = np.array(list(map(float, [row[idx] for idx in indices])))

            # Adjust open price for current bar to match close price of last bar
            if fixOpenPrice and lastClose is not None:
                ppc = lastClose
                if abs(vals[0] - ppc) > 1e-8:
                    countFixed += 1
                    # Update open, and low/high if needed
                    vals[0] = ppc
                    vals[2] = min(vals[2], vals[0])  
                    vals[1] = max(vals[1], vals[0])
            countOut += 1

            # Append values directly to the data array
            data = np.append(data, vals.reshape((5, 1)), axis=1)

            lastClose = vals[3]

    # Print out and return tuple
    print("Read done, got %d rows, %d open prices adjusted" % (countOut, countFixed))
    return Prices(open=  data[0],
                  high=  data[1],
                  low=   data[2],
                  close= data[3],
                  volume=data[4])

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
    return Prices(open=prices.open, 
                  high=relHigh,
                  low=relLow, 
                  close=relClose, 
                  volume=prices.volume)

def price_files(dir_name):
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        result.append(path)
    return result


def load_year_data(year, basedir='data'):
    y = str(year)[-2:]
    result = {}
    for path in glob.glob(os.path.join(basedir, "*_%s*.csv" % y)):
        result[path] = load_relative(path)
    return result
