#!/usr/bin/env python3
""" This script performs training of the code based model """
import pathlib
import argparse

from datasets import load_dataset

from torch import device as hardware

# python3 -m src.train.code_model -p /home/derrick/data/daily_price_data -r test --cuda

# Set random seed for reproducibility
SEED = 42

# Set the directory to save the model
LANGUAGE = "python"
CACHE_DIR = "/home/derrick/data/code/"
SAVES_DIR = pathlib.Path("output")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(      "--cuda", default=False, help="Enable cuda",  action="store_true")
    parser.add_argument("-p", "--path", required=True, help="Directory or file of price data")
    parser.add_argument("-v", "--val",                 help="Validation data, default=path/val/")
    parser.add_argument("-t", "--test",                help="Test data, default=path/test/")
    parser.add_argument("-r", "--run",  required=True, help="Run name")
    args = parser.parse_args()
    device = hardware("cuda" if args.cuda else "cpu")

    # Create output directory
    savesPath = SAVES_DIR / f"{args.run}"
    savesPath.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("bigcode/starcoderdata", cache_dir = CACHE_DIR + LANGUAGE,
                      data_dir=LANGUAGE, split="train")

    # Error out if no data
    if len(ds) == 0:
        raise RuntimeError("No data to train on")
    print(f"For {LANGUAGE}, dataset has {len(ds)} examples")

    sampled_ds = ds.shuffle(seed=42).select(range(10))  # Sample 10 examples
