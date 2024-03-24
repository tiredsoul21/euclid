""" Determines the size of the vocabulary for the dataset. """
import os
import argparse
import pickle

import numpy as np

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", required=True, help="Directory of text data files")
    args = parser.parse_args()

    chunk_size = 1024 * 1024 * 1024 # Process data in 1 GB chunks

    all_token_ids = set()

    train_filename = os.path.join(args.data_dir, 'train.bin')
    with open(train_filename, 'rb') as f:
        i = 0
        while True:
            chunk = np.memmap(f, dtype=np.uint16, mode='r', shape=(chunk_size // 2,))
            if not chunk.any():
                break
            all_token_ids.update(np.unique(chunk))
            del chunk
            print(f"Processed chunk {i}, vocabulary size: {len(all_token_ids)}")

    # Process val.bin in chunks
    val_filename = os.path.join(args.data_dir, 'val.bin')
    with open(val_filename, 'rb') as f:
        while True:
            chunk = np.memmap(f, dtype=np.uint16, mode='r', shape=(chunk_size // 2,))
            if not chunk.any():
                break
            all_token_ids.update(np.unique(chunk))
            del chunk

    # Get the unique token IDs
    vocab_size = len(all_token_ids)

    print(f"Vocabulary size: {vocab_size}")

    meta_data = {
        'vocab_size': vocab_size
    }

    # Save the metadata to a .pkl file
    with open(os.path.join(args.data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta_data, f)
