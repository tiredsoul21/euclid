#!/usr/bin/env python3
import pathlib
import argparse
import gym.wrappers

from lib import data
from lib import environments

SAVES_DIR = pathlib.Path("output")
VAL_DIR = "data/YNDX_150101_151231.csv"
barCount = 50

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Directory or file of price data")
    parser.add_argument("-v", "--val",                 help="Validation data, default=path/val/")
    parser.add_argument("-t", "--test",                help="Test data, default=path/test/")
    parser.add_argument("-r", "--run",  required=True, help="Run name")
    args = parser.parse_args()

    # Create output directory
    savesPath = SAVES_DIR / f"00-{args.run}"
    savesPath.mkdir(parents=True, exist_ok=True)

    # Set data paths
    dataPath = pathlib.Path(args.path)
    dataFolder = dataPath
    
    # If dataPath is a file, use fetch containing directory
    if dataPath.is_file():
        dataFolder = dataPath.parent

    # Set validation path
    if args.val is None:
        valPath = dataFolder / "val"
    else:
        valPath = pathlib.Path(args.val)

    # Set test path
    if args.test is None:
        testPath = dataFolder / "test"
    else:
        testPath = pathlib.Path(args.test)

    # Create Environment
    if dataPath.is_file():
        # Import data from file to dictionary
        index = dataPath.stem
        priceData = {index: data.readCSV(str(dataPath)) }
        env = environments.StocksEnv(priceData, barCount=barCount)
        env._state.barCount = barCount

        env_tst = environments.StocksEnv(priceData, barCount=barCount)
    elif dataPath.is_dir():
        env = environments.StocksEnv.fromDirectory(dataPath, barCount=barCount)
        env_tst = environments.StocksEnv.fromDirectory(dataPath, barCount=barCount)
    else:
        raise RuntimeError("No data to train on")

    # Create validation environment
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    envVal = environments.StocksEnv.fromDirectory(valPath, barCount=barCount)
