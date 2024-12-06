def fetch_lichess_games(username, max_games=20):  # Increase max_games
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max": max_games,
        "moves": True,
        "evals": True,  # Include evaluations in the PGN comments
        "clocks": True,  # Include clock times in the PGN comments
    }
    headers = {"Accept": "application/x-chess-pgn"}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        logging.warning(
            f"Failed to fetch games for user '{username}'. Status code: {response.status_code}"
        )
        return []
    pgn_text = response.text
    games = []
    pgn_io = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break

        # Check if the game contains evaluations
        has_evaluation = False
        node = game
        while node.variations:
            next_node = node.variations[0]
            comment = next_node.comment
            if "%eval" in comment:
                has_evaluation = True
                break
            node = next_node

        if has_evaluation:
            games.append(game)

    logging.info(f"Fetched {len(games)} games with evaluations for user '{username}'.")
    return games


def process_pgn_file(pgn_file_path, max_games=None):
    games = []
    try:
        with open(pgn_file_path, "r", encoding="utf-8") as pgn_file:
            game_counter = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                # Check if the game contains evaluations
                has_evaluation = False
                node = game
                while node.variations:
                    next_node = node.variations[0]
                    comment = next_node.comment
                    if "%eval" in comment:
                        has_evaluation = True
                        break
                    node = next_node

                if has_evaluation:
                    games.append(game)
                    game_counter += 1

                if max_games and game_counter >= max_games:
                    break

        logging.info(
            f"Successfully read {len(games)} games with evaluations from PGN file '{pgn_file_path}'."
        )
    except Exception as e:
        logging.error(f"Failed to read PGN file '{pgn_file_path}': {e}")
    return games


def process_games(games, group_label, engine, max_depth=2):
    all_moves = []
    for game in tqdm(games, desc=f"Processing {group_label} games"):
        try:
            board = game.board()
            game_id = game.headers.get("Site", "Unknown")
            event = game.headers.get("Event", "Unknown")
            date = game.headers.get("UTCDate", "Unknown")
            white = game.headers.get("White", "Unknown")
            black = game.headers.get("Black", "Unknown")
            result = game.headers.get("Result", "Unknown")
            white_elo = safe_int(game.headers.get("WhiteElo", None))
            black_elo = safe_int(game.headers.get("BlackElo", None))
            time_control = game.headers.get("TimeControl", "Unknown")
            
            # Determine which player (if any) has ADHD
            white_has_adhd = white in ADHD_USERNAMES
            black_has_adhd = black in ADHD_USERNAMES

            initial_time, increment, time_category = parse_time_control(time_control)

            node = game
            move_number = 0
            prev_evaluation = None
            current_material = calculate_material(board)
            prev_time_remaining = None
            prev_winning_chances = None

            # Check if the game has evaluations
            has_evaluation = False
            temp_node = node
            while temp_node.variations:
                next_temp_node = temp_node.variations[0]
                comment = next_temp_node.comment
                if "%eval" in comment:
                    has_evaluation = True
                    break
                temp_node = next_temp_node

            if not has_evaluation:
                continue  # Skip game if it doesn't have evaluations

            while node.variations:
                next_node = node.variations[0]
                move = next_node.move
                san = board.san(move)
                move_number += 1
                player = "White" if board.turn else "Black"
                
                # Determine ADHD status for this specific move
                is_adhd_move = (player == "White" and white_has_adhd) or \
                              (player == "Black" and black_has_adhd)

                # Extract clock eval from PGN comments
                comment = next_node.comment
                time_remaining = parse_clock_time(comment)
                evaluation = parse_evaluation(comment)

                # TimeCalc
                if time_remaining is not None and prev_time_remaining is not None:
                    time_spent = prev_time_remaining - time_remaining
                    if time_spent < 0:
                        time_spent = None
                else:
                    time_spent = None
                
                # TimePressure
                under_pressure = is_under_time_pressure(
                    time_remaining=time_remaining,
                    initial_time=initial_time,
                    time_spent=time_spent
                )

                # Calculate winning chances
                winning_chances = eval_winning_chances(evaluation)

                # Calculate winning chances change
                if prev_winning_chances is not None and winning_chances is not None:
                    winning_chances_change = winning_chances - prev_winning_chances
                else:
                    winning_chances_change = None

                # Skip moves without evaluations
                if evaluation is None:
                    board.push(move)
                    node = next_node
                    prev_time_remaining = time_remaining
                    current_material = calculate_material(board)
                    prev_winning_chances = winning_chances
                    continue

                # Apply the move to the board
                board.push(move)

                # Evaluation change
                if prev_evaluation is not None and evaluation is not None:
                    if isinstance(evaluation, str) or isinstance(prev_evaluation, str):
                        eval_change = None
                    else:
                        eval_change = evaluation - prev_evaluation
                else:
                    eval_change = None

                # Error category
                error_category = categorize_error(eval_change)

                # Material difference after the move
                new_material = calculate_material(board)
                material_diff = new_material[player] - current_material[player]

                # Detect sacrifice
                is_sacrifice = material_diff < 0

                # Categorize game phase
                game_phase = categorize_game_phase(move_number)

                # Categorize position complexity based on previous evaluation
                position_complexity = categorize_position_complexity(prev_evaluation)

                # Move condition
                move_condition = "Unknown"

                # Append move data to the list
                move_data = {
                    'GameID': game_id,
                    'Event': event,
                    'Date': date,
                    'White': white,
                    'Black': black,
                    'Result': result,
                    'WhiteElo': white_elo,
                    'BlackElo': black_elo,
                    'TimeControl': time_control,
                    'MoveNumber': move_number,
                    'Player': player,
                    'SAN': san,
                    'Evaluation': evaluation,
                    'EvalChange': eval_change,
                    'ErrorCategory': error_category,
                    'TimeControlCategory': time_category.value,
                    'UnderTimePressure': under_pressure,
                    'InitialTimeSeconds': initial_time,
                    'IncrementSeconds': increment,
                    'TimeRemaining': time_remaining,
                    'TimeSpent': time_spent,
                    'MaterialDiff': material_diff,
                    'IsSacrifice': is_sacrifice,
                    'GamePhase': game_phase,
                    'PositionComplexity': position_complexity,
                    'MoveCondition': move_condition,
                    'IsADHDMove': is_adhd_move,
                    'ADHDPlayer': white if white_has_adhd else (black if black_has_adhd else None),
                    'Group': 'ADHD' if is_adhd_move else 'Control'
                }

                all_moves.append(move_data)

                # Update previous values for next iteration
                prev_evaluation = evaluation
                prev_time_remaining = time_remaining
                current_material = new_material
                prev_winning_chances = winning_chances
                node = next_node

        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            continue

    # Convert the list of moves into a DataFrame and return it
    moves_df = pd.DataFrame(all_moves)
    return moves_df

