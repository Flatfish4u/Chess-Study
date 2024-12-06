def safe_int(value, default=None):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
"""
Setting up Time Functions
"""

def parse_clock_time(comment):
    match = re.search(r'\[%clk (\d+):(\d+):(\d+)\]', comment)  # Adjust regex if needed
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        return hours * 3600 + minutes * 60 + seconds  # Total seconds
    return None

## Determine if a player is under time pressure based on van Harreveld et al. (2007) criteria ---

def is_under_time_pressure(time_remaining, initial_time, time_spent):
    if any(x is None for x in [time_remaining,initial_time, time_spent]):
        return None

    absolute_pressure = time_remaining < 30,
    relative_pressure = (time_remaining / initial_time) < 0.1 if initial_time else False,
    ratio_pressure = (time_spent / time_remaining > 0.3) if time_remaining else False,

    return absolute_pressure or relative_pressure or ratio_pressure

class TimeControlType(Enum):
    CLASSICAL = "Classical"
    RAPID = "Rapid"
    BLITZ = "Blitz"
    BULLET = "Bullet"
    UNKNOWN = "Unknown"

#Parsing and Categorizing Time Control
def parse_time_control(time_control):
    if not time_control or time_control == "unknown":
        return None, None, TimeControlType.UNKNOWN
    try:
        if "+" in time_control:
            base, increment = time_control.split("+")
            base_minutes = int(base)
            increment_seconds = int(increment)
        else:
            base_minutes = int(time_control)
            increment_seconds = 0
        
        initial_time_seconds = base_minutes * 60

        if base_minutes >= 3600:
            category = TimeControlType.CLASSICAL
        elif base_minutes >= 600:
            category = TimeControlType.RAPID
        elif base_minutes >= 180:
            category = TimeControlType.BLITZ
        else:
            category = TimeControlType.BULLET

        return initial_time_seconds, increment_seconds, category

    except (ValueError, TypeError):
        return None, None, TimeControlType.UNKNOWN


def parse_evaluation(comment):
    match = re.search(r'%eval\s([+-]?\d+(\.\d+)?)', comment)
    if match:
        return float(match.group(1))  # Convert to float
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

def categorize_position_complexity(evaluation):
    if evaluation is None:
        return 'Unknown'
    elif abs(evaluation) < 1:
        return 'Balanced'
    elif abs(evaluation) < 3:
        return 'Slight Advantage'
    else:
        return 'Decisive Advantage'

def categorize_move(eval_before, eval_after):
    if eval_before is None or eval_after is None:
        return "Unknown"

    # Define saturation limits in centipawns
    SATURATION_LIMIT = 1000  # Equivalent to a 10-pawn advantage
    MATE_SCORE = 10000       # Arbitrary large value representing mate

    # Calculate evaluation change
    eval_change = eval_after - eval_before

    # Handle mate scores (assuming the engine uses large numbers to indicate mate)
    if abs(eval_after) >= MATE_SCORE:
        if eval_after > 0:
            return "Forced Checkmate (Winning)"
        else:
            return "Forced Checkmate (Losing)"

    # Handle evaluation saturation
    if abs(eval_after) >= SATURATION_LIMIT:
        if eval_after > 0:
            return "Winning Position"
        else:
            return "Losing Position"

    # Categorize the move based on evaluation change
    if eval_change <= -300:
        return "Blunder"
    elif eval_change <= -150:
        return "Mistake"
    elif eval_change <= -50:
        return "Inaccuracy"
    elif eval_change >= 300:
        return "Brilliant Move"
    elif eval_change >= 150:
        return "Great Move"
    elif eval_change >= 50:
        return "Good Move"
    else:
        return "Normal"

def debug_data_pipeline(df, label):
    # Function definition here
    logging.info(f"Debugging {label}")
    # Process the DataFrame or print logs for debugging

def raw_winning_chances(cp):
    MULTIPLIER = -0.00368208
    return 2 / (1 + math.exp(MULTIPLIER * cp)) - 1

def cp_winning_chances(cp):
    cp = max(-1000, min(cp, 1000))
    return raw_winning_chances(cp)

def mate_winning_chances(mate):
    cp = (21 - min(10, abs(mate))) * 100
    signed_cp = cp * (1 if mate > 0 else -1)
    return raw_winning_chances(signed_cp)

def eval_winning_chances(eval_str):
    if eval_str is None:
        return None
    if '#' in str(eval_str):
        # Mate in N moves
        mate_str = str(eval_str).replace('#', '')
        try:
            mate = int(mate_str)
            return mate_winning_chances(mate)
        except ValueError:
            return None
    else:
        try:
            cp = float(eval_str) * 100  # Convert from pawns to centipawns
            return cp_winning_chances(cp)
        except ValueError:
            return None


def safe_int(value, default=None):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def parse_clock_time(comment):
    match = re.search(r'\[%clk (\d+):(\d+):(\d+)\]', comment)  # Adjust regex if needed
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        return hours * 3600 + minutes * 60 + seconds  # Total seconds
    return None


def parse_evaluation(comment):
    match = re.search(r'%eval\s([+-]?[\d.]+|#-?\d+)', comment)
    if match:
        eval_str = match.group(1)
        if '#' in eval_str:
            # Mate in N moves
            return eval_str
        else:
            return float(eval_str)  # Convert to float
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


def categorize_position_complexity(evaluation):
    if evaluation is None:
        return 'Unknown'
    if isinstance(evaluation, str) and '#' in evaluation:
        # Mate positions are considered complex
        return 'Complex'
    elif abs(evaluation) < 1:
        return 'Balanced'
    elif abs(evaluation) < 3:
        return 'Slight Advantage'
    else:
        return 'Decisive Advantage'


def categorize_move(eval_before, eval_after):
    if eval_before is None or eval_after is None:
        return "Unknown"

    # Define saturation limits in centipawns
    SATURATION_LIMIT = 1000  # Equivalent to a 10-pawn advantage
    MATE_SCORE = 10000       # Arbitrary large value representing mate

    # Calculate evaluation change
    eval_change = eval_after - eval_before

    # Handle mate scores (assuming the engine uses large numbers to indicate mate)
    if abs(eval_after) >= MATE_SCORE:
        if eval_after > 0:
            return "Forced Checkmate (Winning)"
        else:
            return "Forced Checkmate (Losing)"

    # Handle evaluation saturation
    if abs(eval_after) >= SATURATION_LIMIT:
        if eval_after > 0:
            return "Winning Position"
        else:
            return "Losing Position"

    # Categorize the move based on evaluation change
    if eval_change <= -300:
        return "Blunder"
    elif eval_change <= -150:
        return "Mistake"
    elif eval_change <= -50:
        return "Inaccuracy"
    elif eval_change >= 300:
        return "Brilliant Move"
    elif eval_change >= 150:
        return "Great Move"
    elif eval_change >= 50:
        return "Good Move"
    else:
        return "Normal"


def debug_data_pipeline(df, label):
    # Function definition here
    logging.info(f"Debugging {label}")
    # Process the DataFrame or print logs for debugging


def raw_winning_chances(cp):
    MULTIPLIER = -0.00368208
    return 2 / (1 + math.exp(MULTIPLIER * cp)) - 1


def cp_winning_chances(cp):
    cp = max(-1000, min(cp, 1000))
    return raw_winning_chances(cp)


def mate_winning_chances(mate):
    cp = (21 - min(10, abs(mate))) * 100
    signed_cp = cp * (1 if mate > 0 else -1)
    return raw_winning_chances(signed_cp)


def eval_winning_chances(evaluation):
    if evaluation is None:
        return None
    if isinstance(evaluation, str) and '#' in evaluation:
        # Mate in N moves
        mate_str = evaluation.replace('#', '')
        try:
            mate = int(mate_str)
            return mate_winning_chances(mate)
        except ValueError:
            return None
    else:
        try:
            cp = float(evaluation) * 100  # Convert from pawns to centipawns
            return cp_winning_chances(cp)
        except ValueError:
            return None
        
