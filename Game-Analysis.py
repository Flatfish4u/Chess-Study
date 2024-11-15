def safe_int(value, default=None):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def parse_clock_time(comment):
    # Extract clock time from comment, e.g., "%clk 1:23:45.678"
    match = re.search(r"%clk\s+([\d:.]+)", comment)
    if match:
        time_str = match.group(1)
        time_parts = [float(part) for part in time_str.split(":")]
        # Weights for hours, minutes, seconds
        weights = [3600, 60, 1]
        weights = weights[-len(time_parts) :]
        seconds = sum(w * t for w, t in zip(weights, time_parts))
        return seconds
    else:
        # Debug statement to check why clock time is not being parsed
        logging.debug(f"Clock time not found in comment: {comment}")
        return None


def parse_evaluation(comment):
    # Extract evaluation from comment, e.g., "%eval 0.34"
    match = re.search(r"%eval\s+([+-]?[0-9]+(\.[0-9]+)?|#-?[0-9]+)", comment)
    if match:
        eval_str = match.group(1)
        if "#" in eval_str:
            # Mate in N moves
            return None
        else:
            return float(eval_str)
    else:
        # Debug statement to check why evaluation is not being parsed
        logging.debug(f"Eval not found in comment: {comment}")
        return None


def categorize_error(eval_change):
    if eval_change is None:
        return "Unknown"
    if eval_change <= -200:
        return "Blunder"
    elif eval_change <= -100:
        return "Mistake"
    elif eval_change <= -50:
        return "Inaccuracy"
    else:
        return "Normal"


def calculate_material(board):
    # Returns material balance for both sides
    material = {"White": 0, "Black": 0}
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,  # King is invaluable, but we set to 0 for simplicity
    }
    for piece_type in piece_values:
        value = piece_values[piece_type]
        material["White"] += len(board.pieces(piece_type, chess.WHITE)) * value
        material["Black"] += len(board.pieces(piece_type, chess.BLACK)) * value
    return material


def categorize_game_phase(move_number):
    if move_number <= 15:
        return "Opening"
    elif move_number <= 30:
        return "Middlegame"
    else:
        return "Endgame"

### This is where things need to be more complex - categorizing position_complexity is not enough -- 
### - There are code functions in python-chess such as 

def categorize_position_complexity(evaluation):
    if evaluation is None:
        return "Unknown"
    elif abs(evaluation) < 1:
        return "Balanced"
    elif abs(evaluation) < 3:
        return "Slight Advantage"
    else:
        return "Decisive Advantage"