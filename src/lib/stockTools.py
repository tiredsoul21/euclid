import numpy as np

from .data import Prices

def moving_avg(prices: Prices, window: int = 10):
    """
    Moving average of the close price
    - Complexity O(n)
    :param window: (int) window size
    :return: moving average
    """
    # Check input parameters
    assert isinstance(prices, Prices)
    assert isinstance(window, int) and window > 0

    # Calculate moving average
    totalSize = prices.close.size
    moving_avgerage = np.zeros(totalSize - window + 1)
    windowSum = np.sum(prices.close[:window])

    #Start from window-th element
    for i in range(window, totalSize):
        moving_avgerage[i - window] = windowSum / window
        windowSum += prices.close[i] - prices.close[i - window]
    
    return moving_avgerage