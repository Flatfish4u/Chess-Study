from config import *                # Import configurations (paths, settings)
import helper_functions as hf        # Import all helper functions with a shorthand (hf)
import chess.pgn                     # Library for reading PGN files
import logging
import pandas as pd                  # For DataFrame manipulations
import chess.engine                  # Stockfish engine interface
from tqdm import tqdm                # Progress bar for processing games
def main():
    print("Beginning Chess Analysis Program")


    print("Local General Population File path:", General_population)
    print("Stockfish Path:", stockfish_path)
    print("LICHESS ADHD Usernames", ADHD_population)


if __name__ == "__main__":
    main()

