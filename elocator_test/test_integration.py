"""Test script for Elocator integration"""
import logging
import chess.pgn
import chess.engine
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import io
import requests
from elocator_api import get_position_complexity_with_retry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def fetch_test_games(username, max_games=2):
    """Fetch a small number of games for testing"""
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max": max_games,
        "moves": True,
        "evals": True,
        "clocks": True,
    }
    headers = {"Accept": "application/x-chess-pgn"}
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            logging.warning(f"Failed to fetch games for {username}")
            return []
        
        pgn = io.StringIO(response.text)
        games = []
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
        return games
    except Exception as e:
        logging.error(f"Error fetching games: {e}")
        return []

def test_position_complexity(username="teoeo"):
    """Test position complexity analysis on a single game"""
    games = fetch_test_games(username, max_games=1)
    if not games:
        logging.error("No games fetched for testing")
        return
    
    game = games[0]
    board = game.board()
    positions = []
    
    # Process first 5 positions
    for i, move in enumerate(game.mainline_moves()):
        if i >= 5:  # Only test first 5 positions
            break
        fen = board.fen()
        complexity = get_position_complexity_with_retry(fen)
        positions.append({
            'fen': fen,
            'complexity': complexity,
            'move_number': i + 1
        })
        board.push(move)
    
    # Display results
    print("\nTest Results:")
    print("-" * 50)
    for pos in positions:
        print(f"Move {pos['move_number']}:")
        print(f"FEN: {pos['fen']}")
        print(f"Complexity: {pos['complexity']}")
        print()
    
    # Plot complexity scores
    complexities = [p['complexity'] for p in positions if p['complexity'] is not None]
    if complexities:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=range(1, len(complexities) + 1), y=complexities)
        plt.title('Position Complexity Scores')
        plt.xlabel('Move Number')
        plt.ylabel('Complexity Score')
        plt.show()

if __name__ == "__main__":
    test_position_complexity()