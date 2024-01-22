#!/usr/bin/env python3
from lib import data

if __name__ == "__main__":
    priceData = None
    priceFiles = data.findFiles('/home/derrick/data/daily_price_data', 'AAPL')
    for file in priceFiles:
        priceData = data.readCSV(file)

