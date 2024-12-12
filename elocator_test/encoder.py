import chess
import numpy as np

def fen_encoder(fen):
    """Convert FEN to 780-dimensional feature vector"""
    # Mirror position if black to move
    board = chess.Board(fen)
    if board.turn == chess.BLACK:
        board = board.mirror()
        fen = board.fen()

    # Initialize board tensor (8x8x12)
    board_tensor = np.zeros((8, 8, 12), dtype=np.uint8)
    
    # Parse position
    position = fen.split()[0]
    ranks = position.split("/")
    
    piece_map = {
        'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
        'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11
    }
    
    # Fill board tensor
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                board_tensor[rank_idx, file_idx, piece_map[char]] = 1
                file_idx += 1
    
    # Add en passant squares (8 dimensions)
    ep_squares = np.zeros(8)
    ep = fen.split()[3]
    if ep != '-':
        file_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        ep_squares[file_map[ep[0]]] = 1
    
    # Add castling rights (4 dimensions)
    castling = np.zeros(4)
    rights = fen.split()[2]
    if 'K' in rights: castling[0] = 1
    if 'Q' in rights: castling[1] = 1
    if 'k' in rights: castling[2] = 1
    if 'q' in rights: castling[3] = 1
    
    # Combine all features (768 + 8 + 4 = 780)
    return np.concatenate([board_tensor.flatten(), ep_squares, castling])