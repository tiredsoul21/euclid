#!/usr/bin/env python3
import pathlib
import argparse

from lib import data
from lib import environments

SAVES_DIR = pathlib.Path("output")
BARS_COUNT = 50

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Directory or file of price data")
    parser.add_argument("-r", "--run",  required=True, help="Run name")
    args = parser.parse_args()

    # Create output directory
    savesPath = SAVES_DIR / f"00-{args.run}"
    savesPath.mkdir(parents=True, exist_ok=True)

    # Set data path
    dataPath = pathlib.Path(args.path)

    # # Read in data
    # priceFiles = data.findFiles(dataPath)
    # print("Found %d files" % len(priceFiles))
    # priceData = data.readCSV(priceFiles[0])

    # Create Environment
    if dataPath.is_file():
        # Import data from file to dictionary
        index = dataPath.stem
        priceData = {index: data.readCSV(str(dataPath)) }
        env = environments.StocksEnv(priceData, bars_count=BARS_COUNT, state_1d=True)
        env_tst = environments.StocksEnv(priceData, bars_count=BARS_COUNT, state_1d=True)
    elif dataPath.is_dir():
        env = environments.StocksEnv.fromDirectory(dataPath, bars_count=BARS_COUNT, state_1d=True)
        env_tst = environments.StocksEnv.fromDirectory(dataPath, bars_count=BARS_COUNT, state_1d=True)
    else:
        raise RuntimeError("No data to train on")

