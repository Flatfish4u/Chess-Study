# Imports
import json
import requests
import pandas as pd
import chess.pgn
import io
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import chess.engine
import sys
import logging

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
)

# Configure plotting style
sns.set(style="whitegrid")

# Replace with the actual path to your general population PGN file
GENERAL_PGN_FILE_PATH = "/Users/benjaminrosales/Desktop/Chess Study/Comparison Games/lichess_db_standard_rated_2017-05.pgn"

# Path to your Stockfish executable
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

# List of ADHD players' usernames (Lichess)
ADHD_USERNAMES = [
    "teoeo",
    "Tobermorey",
    "apostatlet",
    "LovePump1000",
    "Stuntmanandy",
    "Banfy_B",
    "ChessyChesterton12",
    "Yastoon",
    "Timy1976",
    "SonnyDayz11",
    "xiroir",
]