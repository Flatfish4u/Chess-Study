
import logging
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configure plotting style
sns.set(style='whitegrid')

# Path configurations
GENERAL_PGN_FILE_PATH = '/Users/benjaminrosales/Desktop/Chess Study/Comparison Games/lichess_db_standard_rated_2017-05.pgn'
STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'  # **Update this path**

# ADHD Players' usernames
ADHD_USERNAMES = [
    'teoeo', 'Tobermorey', 'apostatlet', 'LovePump1000', 'Stuntmanandy', 
    'Banfy_B', 'ChessyChesterton12', 'Yastoon', 'Timy1976', 'SonnyDayz11', 'xiroir'
]
