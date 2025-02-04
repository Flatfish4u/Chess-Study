# ----------------------- 1. Fetch and Process ADHD Players' Games -----------------------

adhd_games = []
for username in ADHD_USERNAMES:
    logging.info(f"Fetching games for user '{username}'...")
    user_games = fetch_lichess_games(username, max_games=2000)  # Adjust max_games as needed
    adhd_games.extend(user_games)

if not adhd_games:
    logging.warning("No ADHD games fetched. Exiting analysis.")
else:
    # Initialize the chess engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        logging.info(f"Initialized Stockfish engine at '{STOCKFISH_PATH}'.")
    except FileNotFoundError:
        logging.critical(f"Stockfish executable not found at '{STOCKFISH_PATH}'. Please update the path.")
        engine = None
    except Exception as e:
        logging.critical(f"Failed to initialize Stockfish engine: {e}")
        engine = None

    if engine is not None:
        # ----------------------- 2. Process ADHD Players' Games -----------------------
        
        logging.info("Processing ADHD players' games...")
        adhd_moves_df = process_games(adhd_games, group_label='ADHD', engine=engine)
        debug_data_pipeline(adhd_moves_df, "ADHD GAMES PROCESSING")
        
        # ----------------------- 3. Fetch and Process General Population Games -----------------------
        
        logging.info("Fetching general population games...")
        if not os.path.exists(GENERAL_PGN_FILE_PATH):
            logging.error(f"PGN file not found at path: {GENERAL_PGN_FILE_PATH}")
            general_games = []
        else:
            general_games = process_pgn_file(GENERAL_PGN_FILE_PATH, max_games=2000)  # Adjust max_games as needed
        
        if not general_games:
            logging.warning("No General population games to process.")
            general_moves_df = pd.DataFrame()
        else:
            logging.info("Processing general population games...")
            general_moves_df = process_games(general_games, group_label='General', engine=engine)
            debug_data_pipeline(general_moves_df, "GENERAL GAMES PROCESSING")
        
        # ----------------------- 4. Combine Datasets -----------------------

        logging.info("Combining datasets...")
        all_moves_df = pd.concat([adhd_moves_df, general_moves_df], ignore_index=True)
        debug_data_pipeline(all_moves_df, "COMBINED DATASET")

        # ----------------------- 5. Data Cleaning -----------------------

        logging.info("Cleaning data...")
        required_columns = ['TimeSpent', 'Evaluation', 'EvalChange', 'WhiteElo', 'BlackElo']
        # Since we've filtered out moves without evaluations, we can expect 'Evaluation' and 'EvalChange' to be present
        all_moves_df = all_moves_df.dropna(subset=required_columns)

        # Ensure 'IsSacrifice' is boolean
        all_moves_df['IsSacrifice'] = all_moves_df['IsSacrifice'].fillna(False).astype(bool)

        # Convert relevant columns to numeric types
        numeric_columns = ['TimeSpent', 'Evaluation', 'EvalChange', 'WhiteElo', 'BlackElo']
        for col in numeric_columns:
            all_moves_df[col] = pd.to_numeric(all_moves_df[col], errors='coerce')

        # Drop rows with NaNs resulted from non-numeric conversion
        all_moves_df = all_moves_df.dropna(subset=numeric_columns)

        # After cleaning, output the number of moves remaining
        logging.info(f"Total number of moves after cleaning: {len(all_moves_df)}")

        # ----------------------- 6. Statistical Testing -----------------------
        
        logging.info("Performing statistical tests...")
        test_results = []
        
        # ----------------------- 7. Analysis and Plotting -----------------------

        #these functions currently make no sense, and I need to work on plots - what actually I want displayed
    
        
        logging.info("Generating plots and performing statistical tests...")
       # Let's define the missing functions with placeholder plots for each, 
# so the code will run without errors. I'll define these functions one by one.