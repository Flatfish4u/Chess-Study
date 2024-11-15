#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:27:56 2024

@author: benjaminrosales
"""

# %%
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

# %% # ----------------------- Configuration -----------------------


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Configure plotting style
sns.set(style="whitegrid")

# Replace with the actual path to your general population PGN file
GENERAL_PGN_FILE_PATH = "/Users/benjaminrosales/Desktop/Chess Study/Comparison Games/lichess_db_standard_rated_2017-05.pgn"

# Path to your Stockfish executable
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # **Update this path**

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

# %%


def debug_data_pipeline(df, stage_name):
    print(f"\n=== Debugging {stage_name} ===")
    print(f"DataFrame shape: {df.shape}")
    print("\nColumns present:", df.columns.tolist())
    print("\nSample of data (first 5 rows):")
    print(df.head())
    print("\nValue counts for key columns:")
    if "Group" in df.columns:
        print("\nGroup distribution:")
        print(df["Group"].value_counts())
    if "ErrorCategory" in df.columns:
        print("\nErrorCategory distribution:")
        print(df["ErrorCategory"].value_counts())
    print("\nNull values in each column:")
    print(df.isnull().sum())
    print("=" * 50)


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
        # Uncomment the following line to debug clock time parsing
        # print(f"Clock time not found in comment: {comment}")
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
        # Uncomment the following line to debug evaluation parsing
        # print(f"Eval not found in comment: {comment}")
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
        return "Unknown"
    elif abs(evaluation) < 1:
        return "Balanced"
    elif abs(evaluation) < 3:
        return "Slight Advantage"
    else:
        return "Decisive Advantage"


# %% # ----------------------- Statistical Testing Functions -----------------------
def perform_statistical_test(var, data, test_results, test_type="independent_t"):
    # Prepare data
    group1 = data[data["Group"] == "ADHD"][var].dropna()
    group2 = data[data["Group"] == "General"][var].dropna()

    # Check if data is sufficient
    if len(group1) < 10 or len(group2) < 10:
        logging.warning(f"Not enough data to perform statistical test on '{var}'.")
        return

    # Test for normality
    stat1, p1 = stats.shapiro(group1)
    stat2, p2 = stats.shapiro(group2)
    normal = p1 > 0.05 and p2 > 0.05

    # Test for equal variances
    stat_levene, p_levene = stats.levene(group1, group2)
    equal_var = p_levene > 0.05

    # Choose appropriate test
    if normal and equal_var and test_type == "independent_t":
        # Independent T-test
        stat, p = stats.ttest_ind(group1, group2, equal_var=True)
        test_name = "Independent t-test"
    elif normal and not equal_var and test_type == "independent_t":
        # Welch's T-test
        stat, p = stats.ttest_ind(group1, group2, equal_var=False)
        test_name = "Welch's t-test"
    else:
        # Mann-Whitney U Test
        stat, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        test_name = "Mann-Whitney U test"

    test_results.append(
        {"Variable": var, "Test": test_name, "Statistic": stat, "p-value": p}
    )


def perform_chi_squared_test(category_var, data, test_results):
    contingency_table = pd.crosstab(data["Group"], data[category_var])
    if contingency_table.empty or contingency_table.shape[1] == 0:
        logging.warning(f"Contingency table is empty for variable '{category_var}'.")
        return
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    test_results.append(
        {
            "Variable": category_var,
            "Test": "Chi-Squared test",
            "Statistic": chi2,
            "p-value": p,
        }
    )


# %%
# ----------------------- Processing Functions -----------------------


def fetch_lichess_games(username, max_games=1):
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
        games.append(game)
    return games


def process_pgn_file(pgn_file_path, max_games=None):
    games = []
    try:
        with open(pgn_file_path, "r", encoding="utf-8") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
                if max_games and len(games) >= max_games:
                    break
    except Exception as e:
        logging.error(f"Failed to read PGN file: {e}")
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

            node = game
            move_number = 0
            prev_eval = None
            current_material = calculate_material(board)
            prev_time_remaining = None  # Initialize before the loop

            while node.variations:
                next_node = node.variations[0]
                move = next_node.move
                san = board.san(move)
                move_number += 1
                player = "White" if board.turn else "Black"

                # Extract clock time and evaluation from comments
                comment = next_node.comment
                time_remaining = parse_clock_time(comment)
                eval = parse_evaluation(comment)

                # Apply the move to the board
                board.push(move)

                # Calculate time spent
                if time_remaining is not None and prev_time_remaining is not None:
                    time_spent = prev_time_remaining - time_remaining
                    if time_spent < 0:
                        time_spent = None  # Handle clock resets or increments
                else:
                    time_spent = None

                # Eval change
                if prev_eval is not None and eval is not None:
                    eval_change = eval - prev_eval
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
                position_complexity = categorize_position_complexity(prev_eval)

                # Move condition (after move applied)
                move_condition = categorize_move_condition(
                    board.copy(stack=False), move, engine
                )

                move_data = {
                    "GameID": game_id,
                    "Event": event,
                    "Date": date,
                    "White": white,
                    "Black": black,
                    "Result": result,
                    "WhiteElo": white_elo,
                    "BlackElo": black_elo,
                    "TimeControl": time_control,
                    "MoveNumber": move_number,
                    "Player": player,
                    "Move": san,
                    "TimeRemaining": time_remaining,
                    "TimeSpent": time_spent,
                    "Evaluation": eval,
                    "EvalChange": eval_change,
                    "UnderTimePressure": time_remaining is not None
                    and time_remaining < 20,
                    "Group": group_label,
                    "ErrorCategory": error_category,
                    "IsSacrifice": is_sacrifice,
                    "GamePhase": game_phase,
                    "PositionComplexity": position_complexity,
                    "MoveCondition": move_condition,
                }
                all_moves.append(move_data)

                # Update for next iteration
                prev_eval = eval
                prev_time_remaining = time_remaining
                current_material = new_material
                node = next_node
        except Exception as e:
            logging.error(f"Error processing game: {e}")
            continue
    return pd.DataFrame(all_moves)


def categorize_move_condition(board, move, engine):
    """
    Categorizes the move condition as Tactical or Positional based on evaluation changes
    with different move options.
    """
    try:
        # Get the evaluation before the move
        analysis = engine.analyse(board, chess.engine.Limit(depth=2))
        eval_before = analysis["score"].white().score(mate_score=100000)

        # Generate legal moves and evaluate them
        move_scores = []
        for legal_move in board.legal_moves:
            try:
                board.push(legal_move)
                analysis = engine.analyse(board, chess.engine.Limit(depth=2))
                eval_after = analysis["score"].white().score(mate_score=100000)
                board.pop()
                move_scores.append((legal_move, eval_after))
            except Exception as e:
                board.pop()
                logging.error(f"Error analyzing move {legal_move}: {e}")
                continue

        if not move_scores:
            # If move_scores is empty, return 'Unknown' or handle accordingly
            return "Unknown"

        # Sort moves by evaluation
        move_scores.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)

        # Determine if the best move leads to a significant evaluation change
        best_move, best_eval = move_scores[0]
        eval_diff = best_eval - eval_before if eval_before is not None else None

        if eval_diff is not None and abs(eval_diff) > 50:
            return "Tactical"
        else:
            return "Positional"
    except Exception as e:
        logging.error(f"Error categorizing move condition: {e}")
        return "Unknown"


# %%
# ----------------------- Analysis and Plotting Functions -----------------------


def plot_performance_under_time_pressure(df, test_results):
    # Filter moves under time pressure
    df_pressure = df[df["UnderTimePressure"]]
    if df_pressure.empty:
        logging.warning("No moves under time pressure to analyze.")
        return

    # Ensure 'EvalChange' is numeric
    df_pressure["EvalChange"] = pd.to_numeric(
        df_pressure["EvalChange"], errors="coerce"
    )
    df_pressure = df_pressure.dropna(subset=["EvalChange"])

    # Calculate average EvalChange under time pressure
    avg_eval_change = df_pressure.groupby("Group")["EvalChange"].mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Group", y="EvalChange", data=avg_eval_change, palette="Set2")
    plt.title("Average Evaluation Change Under Time Pressure")
    plt.ylabel("Average Evaluation Change (centipawns)")
    plt.xlabel("Group")
    plt.show()

    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Group", y="EvalChange", data=df_pressure, palette="Set2", inner="quartile"
    )
    plt.title("Distribution of Evaluation Change Under Time Pressure")
    plt.ylabel("Evaluation Change (centipawns)")
    plt.xlabel("Group")
    plt.show()

    # Statistical Test
    perform_statistical_test("EvalChange", df_pressure, test_results)


def plot_accuracy_vs_time(df, test_results):
    # Ensure 'TimeSpent' and 'EvalChange' are numeric
    df["TimeSpent"] = pd.to_numeric(df["TimeSpent"], errors="coerce")
    df["EvalChange"] = pd.to_numeric(df["EvalChange"], errors="coerce")
    df = df.dropna(subset=["TimeSpent", "EvalChange"])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="TimeSpent", y="EvalChange", hue="Group", data=df, alpha=0.3, palette="Set1"
    )
    sns.lineplot(
        x="TimeSpent",
        y="EvalChange",
        hue="Group",
        data=df,
        estimator="mean",
        palette="Set1",
    )
    plt.title("Move Accuracy Relative to Move Time")
    plt.xlabel("Time Spent on Move (seconds)")
    plt.ylabel("Evaluation Change (centipawns)")
    plt.legend(title="Group")
    plt.show()

    # Statistical Test
    perform_statistical_test("EvalChange", df, test_results)


def plot_error_rate(df, test_results):
    error_counts = (
        df.groupby(["Group", "ErrorCategory"]).size().reset_index(name="Count")
    )
    total_counts = (
        df.groupby("Group")["ErrorCategory"].count().reset_index(name="Total")
    )
    error_counts = error_counts.merge(total_counts, on="Group")
    error_counts["Percentage"] = (error_counts["Count"] / error_counts["Total"]) * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="ErrorCategory",
        y="Percentage",
        hue="Group",
        data=error_counts,
        palette="Set3",
    )
    plt.title("Error Rate Comparison")
    plt.ylabel("Percentage of Moves (%)")
    plt.xlabel("Error Category")
    plt.legend(title="Group")
    plt.xticks(rotation=45)
    plt.show()

    # Chi-Squared Test
    perform_chi_squared_test("ErrorCategory", df, test_results)


def plot_time_management(df, test_results):
    # Ensure 'TimeSpent' is numeric
    df["TimeSpent"] = pd.to_numeric(df["TimeSpent"], errors="coerce")
    df = df.dropna(subset=["TimeSpent"])

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=df,
        x="TimeSpent",
        hue="Group",
        common_norm=False,
        fill=True,
        alpha=0.5,
        palette="Set1",
    )
    plt.title("Time Management Patterns")
    plt.xlabel("Time Spent on Move (seconds)")
    plt.ylabel("Density")
    plt.legend(title="Group")
    plt.show()

    # Box Plot Comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Group", y="TimeSpent", data=df, palette="Set2")
    plt.title("Time Spent per Move by Group")
    plt.ylabel("Time Spent on Move (seconds)")
    plt.xlabel("Group")
    plt.show()

    # Statistical Test
    perform_statistical_test("TimeSpent", df, test_results)


def stratify_by_elo(df, test_results):
    # Create ELO categories
    def categorize_elo(elo):
        if elo < 1400:
            return "<1400"
        elif elo < 2000:
            return "1400-1999"
        else:
            return "2000+"

    df["EloCategory"] = df.apply(
        lambda x: (
            categorize_elo(x["WhiteElo"])
            if x["Player"] == "White"
            else categorize_elo(x["BlackElo"])
        ),
        axis=1,
    )
    df = df.dropna(subset=["EloCategory"])

    # Ensure 'EvalChange' and 'TimeSpent' are numeric
    df["EvalChange"] = pd.to_numeric(df["EvalChange"], errors="coerce")
    df["TimeSpent"] = pd.to_numeric(df["TimeSpent"], errors="coerce")
    df = df.dropna(subset=["EvalChange", "TimeSpent"])

    # Plot accuracy vs time for each ELO category
    g = sns.FacetGrid(
        df,
        col="EloCategory",
        hue="Group",
        col_wrap=3,
        height=4,
        sharey=False,
        palette="Set1",
    )
    g.map_dataframe(sns.scatterplot, x="TimeSpent", y="EvalChange", alpha=0.3)
    g.add_legend(title="Group")
    g.set_axis_labels("Time Spent on Move (seconds)", "Evaluation Change (centipawns)")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Accuracy vs Time Spent by ELO Category")
    plt.show()

    # Box Plot by ELO Category
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="EloCategory", y="EvalChange", hue="Group", data=df, palette="Set2")
    plt.title("Evaluation Change by ELO Category and Group")
    plt.ylabel("Evaluation Change (centipawns)")
    plt.xlabel("ELO Category")
    plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()

    # Statistical tests within each ELO category
    for elo in df["EloCategory"].unique():
        subset = df[df["EloCategory"] == elo]
        perform_statistical_test("EvalChange", subset, test_results)


# %%
# ----------------------- Main Execution -----------------------


def main():
    # ----------------------- 1. Fetch and Process ADHD Players' Games -----------------------

    adhd_games = []
    for username in ADHD_USERNAMES:
        logging.info(f"Fetching games for user '{username}'...")
        user_games = fetch_lichess_games(
            username, max_games=2
        )  # Adjust max_games as needed
        adhd_games.extend(user_games)

    if not adhd_games:
        logging.warning("No ADHD games fetched. Exiting analysis.")
        return

    # Initialize the chess engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        logging.info(f"Initialized Stockfish engine at '{STOCKFISH_PATH}'.")
    except FileNotFoundError:
        logging.critical(
            f"Stockfish executable not found at '{STOCKFISH_PATH}'. Please update the path."
        )
        return
    except Exception as e:
        logging.critical(f"Failed to initialize Stockfish engine: {e}")
        return

    # ----------------------- 2. Process ADHD Players' Games -----------------------

    logging.info("Processing ADHD players' games...")
    adhd_moves_df = process_games(adhd_games, group_label="ADHD", engine=engine)
    debug_data_pipeline(adhd_moves_df, "ADHD GAMES PROCESSING")

    # ----------------------- 3. Fetch and Process General Population Games -----------------------

    logging.info("Fetching general population games...")
    general_games = process_pgn_file(
        GENERAL_PGN_FILE_PATH, max_games=20
    )  # Adjust max_games as needed

    if not general_games:
        logging.warning("No General population games to process.")
        general_moves_df = pd.DataFrame()
    else:
        logging.info("Processing general population games...")
        general_moves_df = process_games(
            general_games, group_label="General", engine=engine
        )

    # ----------------------- 4. Combine Datasets -----------------------

    logging.info("Combining datasets...")
    all_moves_df = pd.concat([adhd_moves_df, general_moves_df], ignore_index=True)

    # ----------------------- 5. Data Cleaning -----------------------

    logging.info("Cleaning data...")
    required_columns = ["TimeSpent", "Evaluation", "EvalChange", "WhiteElo", "BlackElo"]
    all_moves_df = all_moves_df.dropna(subset=required_columns)

    # Ensure 'IsSacrifice' is boolean
    all_moves_df["IsSacrifice"] = all_moves_df["IsSacrifice"].fillna(False).astype(bool)

    # Convert relevant columns to numeric types
    numeric_columns = ["TimeSpent", "Evaluation", "EvalChange", "WhiteElo", "BlackElo"]
    for col in numeric_columns:
        all_moves_df[col] = pd.to_numeric(all_moves_df[col], errors="coerce")

    # Drop rows with NaNs resulted from non-numeric conversion
    all_moves_df = all_moves_df.dropna(subset=numeric_columns)

    # ----------------------- 6. Statistical Testing -----------------------

    logging.info("Performing statistical tests...")
    test_results = []

    # ----------------------- 7. Analysis and Plotting -----------------------

    logging.info("Generating plots and performing statistical tests...")
    plot_performance_under_time_pressure(all_moves_df, test_results)
    plot_accuracy_vs_time(all_moves_df, test_results)
    plot_error_rate(all_moves_df, test_results)
    plot_time_management(all_moves_df, test_results)
    stratify_by_elo(all_moves_df, test_results)

    # ----------------------- 8. Display Statistical Test Results -----------------------

    logging.info(
        "\n----------------------- Statistical Test Results -----------------------\n"
    )
    results_df = pd.DataFrame(test_results)

    if not results_df.empty:
        # Apply Bonferroni correction for multiple comparisons
        num_tests = len(results_df)
        results_df["Adjusted p-value"] = results_df["p-value"] * num_tests
        results_df["Adjusted p-value"] = results_df["Adjusted p-value"].apply(
            lambda x: min(x, 1.0)
        )

        # Determine significance after correction
        results_df["Significant"] = results_df["Adjusted p-value"] < 0.05

        # Display the results
        print(
            results_df[
                [
                    "Variable",
                    "Test",
                    "Statistic",
                    "p-value",
                    "Adjusted p-value",
                    "Significant",
                ]
            ]
        )

        logging.info(
            "\nNote: p-values have been adjusted using the Bonferroni correction for multiple comparisons.\n"
        )
    else:
        logging.info("No statistical tests were performed.")

    # ----------------------- Cleanup -----------------------

    # Close the chess engine
    engine.quit()

    logging.info("Analysis complete.")


# ----------------------- Execute the Script -----------------------
if __name__ == "__main__":
    main()
