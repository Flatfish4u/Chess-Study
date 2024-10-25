#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chess Analysis Script
=====================

This script analyzes chess games of ADHD players versus the general population.
It addresses the following questions:
1. In what kinds of positions do ADHD players take longer to make moves?
2. Under certain move conditions, do they perform better or worse?

Author: Benjamin Rosales
Date: [Date]

"""

# ----------------------- Import Libraries -----------------------
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

# ----------------------- Configuration -----------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Configure plotting style
sns.set(style='whitegrid')

# Replace with the actual path to your general population PGN file
GENERAL_PGN_FILE_PATH = '/Users/benjaminrosales/Desktop/Chess Study/Comparison Games/lichess_db_standard_rated_2017-05.pgn'  # **Update this path**

# Path to your Stockfish executable
STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'  # **Update this path**

# List of ADHD players' usernames (Lichess)
ADHD_USERNAMES = ['teoeo', 'Tobermorey', 'apostatlet', 'LovePump1000', 'Stuntmanandy', 
                 'Banfy_B', 'ChessyChesterton12', 'Yastoon', 'Timy1976', 'SonnyDayz11', 'xiroir']

# ----------------------- Helper Functions -----------------------

# Define piece values for material calculation
PIECE_VALUES = {
    'P': 1,
    'N': 3,
    'B': 3,
    'R': 5,
    'Q': 9,
    'K': 0
}

def safe_int(value, default=None):
    """
    Safely converts a value to an integer.
    Returns the default value if conversion fails.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def parse_clock_time(comment):
    """
    Parses clock time strings in the format of '%clk HH:MM:SS' or '%clk MM:SS'.
    Returns time in seconds.
    """
    # Pattern for HH:MM:SS
    time_match = re.search(r'\[%clk (\d+):(\d{2}):(\d{2})\]', comment)
    if time_match:
        hours, minutes, seconds = map(int, time_match.groups())
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    else:
        # Pattern for MM:SS
        time_match = re.search(r'\[%clk (\d{1,2}):(\d{2})\]', comment)
        if time_match:
            minutes, seconds = map(int, time_match.groups())
            total_seconds = minutes * 60 + seconds
            return total_seconds
    return None

def parse_evaluation(comment):
    """
    Parses evaluation strings in the format of '%eval N.N' or '%eval #N'.
    Returns evaluation in centipawns.
    """
    eval_match = re.search(r'\[%eval ([^\]]+)\]', comment)
    if eval_match:
        eval_str = eval_match.group(1)
        if '#' in eval_str:
            # Mate in N moves
            try:
                mate_in = int(eval_str.replace('#', ''))
                eval_value = np.sign(mate_in) * 10000  # Arbitrary large value for mate
            except ValueError:
                eval_value = None
        else:
            try:
                eval_value = float(eval_str) * 100  # Convert to centipawns
            except ValueError:
                eval_value = None
        return eval_value
    return None

def categorize_error(eval_change):
    """
    Categorizes the error based on evaluation change.
    """
    if eval_change is None:
        return 'Unknown'
    if eval_change <= -300:
        return 'Blunder'
    elif eval_change <= -100:
        return 'Mistake'
    elif eval_change <= -50:
        return 'Inaccuracy'
    else:
        return 'Good Move'

def calculate_material(board):
    """
    Calculates the material balance for both White and Black.
    Returns a tuple (white_material, black_material).
    """
    white_material = 0
    black_material = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.symbol().upper(), 0)
        if piece.color:
            white_material += value
        else:
            black_material += value
    return white_material, black_material

def categorize_game_phase(move_number):
    """
    Categorizes the game phase based on move number.
    """
    if move_number <= 10:
        return 'Opening'
    elif move_number <= 30:
        return 'Middlegame'
    else:
        return 'Endgame'

def categorize_position_complexity(evaluation):
    """
    Categorizes position complexity based on engine evaluation.
    """
    if evaluation is None:
        return 'Unknown'
    elif evaluation < -300 or evaluation > 300:
        return 'High Complexity'
    elif -100 <= evaluation <= 100:
        return 'Low Complexity'
    else:
        return 'Medium Complexity'

def categorize_move_condition(board, move, engine):
    """
    Categorizes the move condition as 'Tactical' or 'Positional'.
    A move is considered 'Tactical' if it results in a significant evaluation change.
    """
    try:
        # Make the move on a copy of the board
        board_copy = board.copy()
        board_copy.push(move)
        
        # Analyze the new position with limited time for efficiency
        result = engine.analyse(board_copy, chess.engine.Limit(time=0.1))
        score = result['score'].relative
        
        if score.is_mate():
            new_eval = score.mate()
            if new_eval is not None:
                new_eval = np.sign(new_eval) * 10000
        else:
            new_eval = score.score(mate_score=10000)
        
        if new_eval is None:
            return 'Unknown'
        elif abs(new_eval) > 300:
            return 'Tactical'
        else:
            return 'Positional'
    except Exception as e:
        logging.error(f"Error categorizing move condition: {e}")
        return 'Unknown'

def detect_sacrifice(prev_material, current_material, eval_change):
    """
    Detects if the move was a sacrifice based on material loss and evaluation change.
    """
    white_prev, black_prev = prev_material
    white_curr, black_curr = current_material

    # Define a threshold for material loss to consider
    MATERIAL_LOSS_THRESHOLD = 1  # At least a pawn

    if eval_change is None:
        return False

    if eval_change > 50 and (white_prev - white_curr >= MATERIAL_LOSS_THRESHOLD or black_prev - black_curr >= MATERIAL_LOSS_THRESHOLD):
        return True
    return False

def perform_statistical_test(var, data, test_results, test_type='independent_t'):
    """
    Performs a statistical test between ADHD and General groups for a given variable.
    Appends the result to test_results list.
    
    Parameters:
    - var (str): The variable/column name to test.
    - data (pd.DataFrame): The DataFrame containing the data.
    - test_results (list): A list to store the test results.
    - test_type (str): The type of test to perform ('independent_t' or 'chi_squared').
    """
    group1 = data[data['Group'] == 'ADHD'][var]
    group2 = data[data['Group'] == 'General'][var]
    
    # Check if groups have enough data
    if len(group1) < 3 or len(group2) < 3:
        logging.warning(f"Not enough data to perform test for variable '{var}'.")
        return
    
    if test_type == 'independent_t':
        # Check normality
        sample_size = min(500, len(group1), len(group2))  # Shapiro-Wilk has a max sample size
        try:
            stat1, p1 = stats.shapiro(group1.sample(sample_size, random_state=1))
            stat2, p2 = stats.shapiro(group2.sample(sample_size, random_state=1))
            normal1 = p1 > 0.05
            normal2 = p2 > 0.05
        except Exception as e:
            logging.error(f"Shapiro-Wilk test failed for variable '{var}': {e}")
            normal1, normal2 = False, False
        
        # Check variance equality
        try:
            stat_levene, p_levene = stats.levene(group1, group2)
            equal_var = p_levene > 0.05
        except Exception as e:
            logging.error(f"Levene's test failed for variable '{var}': {e}")
            equal_var = False
        
        if normal1 and normal2:
            # Use Independent Samples t-test
            try:
                stat, p = stats.ttest_ind(group1, group2, equal_var=equal_var)
                test_used = 'Independent t-test'
            except Exception as e:
                logging.error(f"t-test failed for variable '{var}': {e}")
                return
        else:
            # Use Mann-Whitney U test
            try:
                stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                test_used = 'Mann-Whitney U test'
            except Exception as e:
                logging.error(f"Mann-Whitney U test failed for variable '{var}': {e}")
                return
        
        # Store results
        test_results.append({
            'Variable': var,
            'Test': test_used,
            'Statistic': stat,
            'p-value': p
        })
    
    elif test_type == 'chi_squared':
        # For categorical variables
        try:
            contingency_table = pd.crosstab(data['Group'], data[var])
            stat, p, dof, expected = stats.chi2_contingency(contingency_table)
            test_results.append({
                'Variable': var,
                'Test': 'Chi-Squared test',
                'Statistic': stat,
                'p-value': p
            })
        except Exception as e:
            logging.error(f"Chi-Squared test failed for variable '{var}': {e}")
            return
    else:
        logging.error(f"Unknown test type '{test_type}' for variable '{var}'.")
        return

def perform_chi_squared_test(category_var, data, test_results):
    """
    Performs Chi-Squared test between ADHD and General groups for a given categorical variable.
    Appends the result to test_results list.
    """
    perform_statistical_test(category_var, data, test_results, test_type='chi_squared')

# ----------------------- Processing Functions -----------------------

def fetch_lichess_games(username, max_games=2000):
    """
    Fetches games from Lichess API for a given username.
    
    Parameters:
    - username (str): Lichess username.
    - max_games (int): Maximum number of games to fetch.
    
    Returns:
    - list: List of chess.pgn.Game objects.
    """
    url = f'https://lichess.org/api/games/user/{username}'
    headers = {
        'Accept': 'application/x-chess-pgn',
    }
    params = {
        'max': max_games,
        'clocks': True,
        'evals': True,
        'opening': False,
        'moves': True,
    }

    try:
        response = requests.get(url, params=params, headers=headers, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request exception for {username}: {e}")
        return []

    games = []

    pgn_content = response.content.decode('utf-8').strip()
    pgn_io = io.StringIO(pgn_content)
    game_counter = 0
    pbar = tqdm(total=max_games, desc=f'Fetching games for {username}', leave=False)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        games.append(game)
        game_counter += 1
        pbar.update(1)
        if game_counter >= max_games:
            break
    pbar.close()
    logging.info(f"Fetched {len(games)} games for user '{username}'.")
    return games

def process_pgn_file(pgn_file_path, max_games=None):
    """
    Processes a PGN file and extracts games up to max_games.
    
    Parameters:
    - pgn_file_path (str): Path to the PGN file.
    - max_games (int, optional): Maximum number of games to process.
    
    Returns:
    - list: List of chess.pgn.Game objects.
    """
    games = []
    try:
        with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file:
            game_counter = 0
            pbar = tqdm(total=max_games if max_games else None, desc='Processing general population games', leave=False)
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
                game_counter += 1
                pbar.update(1)
                if max_games and game_counter >= max_games:
                    break
            pbar.close()
        logging.info(f"Processed {len(games)} games from '{pgn_file_path}'.")
    except FileNotFoundError:
        logging.error(f"PGN file not found at '{pgn_file_path}'. Please update the path.")
    except Exception as e:
        logging.error(f"Error processing PGN file: {e}")
    return games

def process_games(games, group_label, engine, max_depth=15):
    """
    Processes a list of chess games and extracts move-level data, including sacrifices,
    position complexity, and move conditions.
    
    Parameters:
    - games (list): List of chess.pgn.Game objects.
    - group_label (str): Label for the group ('ADHD' or 'General').
    - engine (chess.engine.SimpleEngine): The chess engine instance.
    - max_depth (int): Maximum depth for Stockfish analysis.
    
    Returns:
    - pd.DataFrame: DataFrame containing move data.
    """
    all_moves = []
    for game in tqdm(games, desc=f'Processing {group_label} games'):
        # Exclude chess variants
        if game.headers.get('Variant', 'Standard') != 'Standard':
            continue

        # Get player ratings using safe_int
        white_elo = safe_int(game.headers.get('WhiteElo'))
        black_elo = safe_int(game.headers.get('BlackElo'))

        # Optionally skip games with unknown ELO ratings
        if white_elo is None or black_elo is None:
            continue

        board = game.board()
        node = game

        prev_time = {True: None, False: None}  # {White's time, Black's time}
        prev_eval = None
        move_number = 0

        # Initialize material balance
        prev_material = calculate_material(board)

        while node.variations:
            next_node = node.variations[0]
            move = next_node.move
            san = board.san(move)
            move_number += 1
            player = 'White' if board.turn else 'Black'

            # Extract clock time and evaluation from comments
            comment = next_node.comment

            time_remaining = parse_clock_time(comment)
            eval_score = parse_evaluation(comment)

            time_spent = None
            if prev_time[board.turn] is not None and time_remaining is not None:
                time_spent = prev_time[board.turn] - time_remaining
                if time_spent < 0:
                    time_spent = None  # Handle clock increments or errors

            eval_change = None
            if prev_eval is not None and eval_score is not None:
                eval_change = eval_score - prev_eval

            under_time_pressure = time_remaining is not None and time_remaining < 20

            # Calculate current material balance after the move
            board.push(move)
            current_material = calculate_material(board)

            # Determine material difference for the player
            if player == 'White':
                material_diff = current_material[0] - prev_material[0]
            else:
                material_diff = current_material[1] - prev_material[1]

            # Detect sacrifice
            is_sacrifice = False
            if material_diff < -1:  # Player lost more than a pawn
                if eval_change is not None and eval_change > 50:  # Position improved despite material loss
                    is_sacrifice = True

            # Categorize game phase
            game_phase = categorize_game_phase(move_number)

            # Categorize position complexity before the move
            position_complexity = categorize_position_complexity(prev_eval)

            # Categorize move condition
            move_condition = categorize_move_condition(board.copy(stack=False), move, engine)

            move_data = {
                'GameID': game.headers.get('Site', ''),
                'Event': game.headers.get('Event', ''),
                'Date': game.headers.get('UTCDate', ''),
                'White': game.headers.get('White', ''),
                'Black': game.headers.get('Black', ''),
                'Result': game.headers.get('Result', ''),
                'WhiteElo': white_elo,
                'BlackElo': black_elo,
                'TimeControl': game.headers.get('TimeControl', ''),
                'MoveNumber': move_number,
                'Player': player,
                'Move': san,
                'TimeRemaining': time_remaining,
                'TimeSpent': time_spent,
                'Evaluation': eval_score,
                'EvalChange': eval_change,
                'UnderTimePressure': under_time_pressure,
                'Group': group_label,
                'ErrorCategory': categorize_error(eval_change),
                'IsSacrifice': is_sacrifice,
                'GamePhase': game_phase,
                'PositionComplexity': position_complexity,
                'MoveCondition': move_condition
            }

            all_moves.append(move_data)

            prev_time[board.turn] = time_remaining
            prev_eval = eval_score
            prev_material = current_material  # Removed .copy()

            node = next_node

    return pd.DataFrame(all_moves)

# ----------------------- Analysis and Plotting Functions -----------------------

def plot_performance_under_time_pressure(df, test_results):
    """
    Plots average evaluation change under time pressure for both groups.
    Also performs statistical testing.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing all moves.
    - test_results (list): List to store statistical test results.
    """
    # Only consider moves under time pressure
    df_pressure = df[df['UnderTimePressure']]

    if df_pressure.empty:
        logging.info("No moves under time pressure to plot.")
        return

    # Ensure 'EvalChange' is numeric
    df_pressure['EvalChange'] = pd.to_numeric(df_pressure['EvalChange'], errors='coerce')

    # Drop NaNs resulted from non-numeric 'EvalChange'
    df_pressure = df_pressure.dropna(subset=['EvalChange'])

    # Calculate average evaluation change under time pressure for each group
    avg_eval_change = df_pressure.groupby('Group')['EvalChange'].mean().reset_index()

    # Plotting - Bar Plot
    plt.figure(figsize=(8,6))
    ax = sns.barplot(x='Group', y='EvalChange', data=avg_eval_change, palette='Set2')
    plt.title('Average Evaluation Change Under Time Pressure (<20s)')
    plt.ylabel('Average Evaluation Change (centipawns)')
    plt.xlabel('Group')

    # Perform statistical test
    perform_statistical_test('EvalChange', df_pressure, test_results, test_type='independent_t')

    # Retrieve the last test result
    if test_results:
        last_test = test_results[-1]
        p_val = last_test['p-value']
        # Apply Bonferroni correction later
        # Annotate significance
        if p_val * len(test_results) < 0.05:
            y_max = avg_eval_change['EvalChange'].max()
            ax.text(0.5, y_max + 50, '*', ha='center', va='bottom', fontsize=20)

    plt.show()

    # Violin Plot to show distribution
    plt.figure(figsize=(10,6))
    sns.violinplot(x='Group', y='EvalChange', data=df_pressure, palette='Set2', inner='quartile')
    plt.title('Distribution of Evaluation Change Under Time Pressure (<20s)')
    plt.ylabel('Evaluation Change (centipawns)')
    plt.xlabel('Group')
    plt.show()

def plot_accuracy_vs_time(df, test_results):
    """
    Plots move accuracy relative to move time, comparing both groups.
    Also performs statistical testing.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing all moves.
    - test_results (list): List to store statistical test results.
    """
    # Ensure 'TimeSpent' and 'EvalChange' are numeric
    df['TimeSpent'] = pd.to_numeric(df['TimeSpent'], errors='coerce')
    df['EvalChange'] = pd.to_numeric(df['EvalChange'], errors='coerce')

    # Drop rows with NaNs in 'TimeSpent' or 'EvalChange'
    df = df.dropna(subset=['TimeSpent', 'EvalChange'])

    plt.figure(figsize=(10,6))
    sns.scatterplot(x='TimeSpent', y='EvalChange', hue='Group', data=df, alpha=0.3, palette='Set1')
    sns.lineplot(x='TimeSpent', y='EvalChange', hue='Group', data=df, ci=None, estimator='mean', palette='Set1')
    plt.title('Move Accuracy Relative to Move Time')
    plt.xlabel('Time Spent on Move (seconds)')
    plt.ylabel('Evaluation Change (centipawns)')
    plt.legend(title='Group')
    plt.show()

    # Regression plot
    sns.lmplot(x='TimeSpent', y='EvalChange', hue='Group', data=df, height=6, aspect=1.5, palette='Set1', scatter_kws={'alpha':0.2, 's':10})
    plt.title('Move Accuracy Relative to Move Time (Regression)')
    plt.xlabel('Time Spent on Move (seconds)')
    plt.ylabel('Evaluation Change (centipawns)')
    plt.show()

    # Perform statistical test
    perform_statistical_test('EvalChange', df, test_results, test_type='independent_t')

def plot_error_rate(df, test_results):
    """
    Plots error rate comparison between both groups.
    Also performs Chi-Squared testing.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing all moves.
    - test_results (list): List to store statistical test results.
    """
    # Calculate error counts
    error_counts = df.groupby(['Group', 'ErrorCategory']).size().reset_index(name='Count')

    # Ensure 'Count' is numeric
    error_counts['Count'] = pd.to_numeric(error_counts['Count'], errors='coerce')

    # Normalize counts to get percentages
    total_counts = df.groupby('Group')['ErrorCategory'].count().reset_index(name='Total')

    # Ensure 'Total' is numeric
    total_counts['Total'] = pd.to_numeric(total_counts['Total'], errors='coerce')

    error_counts = error_counts.merge(total_counts, on='Group')
    error_counts['Percentage'] = (error_counts['Count'] / error_counts['Total']) * 100

    # Drop rows with NaNs resulted from non-numeric conversion
    error_counts = error_counts.dropna(subset=['Percentage'])

    # Sort ErrorCategory for consistent plotting
    error_order = ['Blunder', 'Mistake', 'Inaccuracy', 'Good Move', 'Unknown']
    error_counts['ErrorCategory'] = pd.Categorical(error_counts['ErrorCategory'], categories=error_order, ordered=True)

    # Plotting
    plt.figure(figsize=(10,6))
    sns.barplot(x='ErrorCategory', y='Percentage', hue='Group', data=error_counts, palette='Set3')
    plt.title('Error Rate Comparison')
    plt.ylabel('Percentage of Moves (%)')
    plt.xlabel('Error Category')
    plt.legend(title='Group')
    plt.xticks(rotation=45)
    plt.show()

    # Perform Chi-Squared test
    perform_chi_squared_test('ErrorCategory', df, test_results)

def plot_time_management(df, test_results):
    """
    Plots time management patterns for both groups.
    Also performs statistical testing.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing all moves.
    - test_results (list): List to store statistical test results.
    """
    # Ensure 'TimeSpent' is numeric
    df['TimeSpent'] = pd.to_numeric(df['TimeSpent'], errors='coerce')

    # Drop rows with NaNs in 'TimeSpent'
    df = df.dropna(subset=['TimeSpent'])

    plt.figure(figsize=(10,6))
    sns.kdeplot(data=df, x='TimeSpent', hue='Group', common_norm=False, fill=True, alpha=0.5, palette='Set1')
    plt.title('Time Management Patterns')
    plt.xlabel('Time Spent on Move (seconds)')
    plt.ylabel('Density')
    plt.legend(title='Group')
    plt.show()

    # Box Plot Comparison
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Group', y='TimeSpent', data=df, palette='Set2')
    plt.title('Time Spent per Move by Group')
    plt.ylabel('Time Spent on Move (seconds)')
    plt.xlabel('Group')
    plt.show()

    # Perform statistical test
    perform_statistical_test('TimeSpent', df, test_results, test_type='independent_t')

def stratify_by_elo(df, test_results):
    """
    Stratifies data by ELO categories and compares both groups within each category.
    Also performs statistical testing within each ELO category.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing all moves.
    - test_results (list): List to store statistical test results.
    """
    # Create ELO categories (simplified to 3 groups)
    def categorize_elo(elo):
        if elo < 1400:
            return '<1400'
        elif elo < 2000:
            return '1400-1999'
        else:
            return '2000+'

    df['EloCategory'] = df.apply(lambda x: categorize_elo(x['WhiteElo']) if x['Player'] == 'White' else categorize_elo(x['BlackElo']), axis=1)
    
    # Exclude rows with missing EloCategory
    df = df.dropna(subset=['EloCategory'])

    # Ensure 'EvalChange' is numeric
    df['EvalChange'] = pd.to_numeric(df['EvalChange'], errors='coerce')

    # Drop rows with NaNs in 'EvalChange'
    df = df.dropna(subset=['EvalChange'])

    # Plot accuracy vs time for each ELO category
    g = sns.FacetGrid(df, col='EloCategory', hue='Group', col_wrap=3, height=4, sharey=False, palette='Set1')
    g.map_dataframe(sns.scatterplot, x='TimeSpent', y='EvalChange', alpha=0.3)
    g.add_legend(title='Group')
    g.set_axis_labels('Time Spent on Move (seconds)', 'Evaluation Change (centipawns)')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Accuracy vs Time Spent by ELO Category')
    plt.show()

    # Box Plot by ELO Category
    plt.figure(figsize=(14,8))
    sns.boxplot(x='EloCategory', y='EvalChange', hue='Group', data=df, palette='Set2')
    plt.title('Evaluation Change by ELO Category and Group')
    plt.ylabel('Evaluation Change (centipawns)')
    plt.xlabel('ELO Category')
    plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Perform statistical tests within each ELO category
    elo_categories = df['EloCategory'].unique()
    for elo in elo_categories:
        subset = df[df['EloCategory'] == elo]
        if subset['Group'].nunique() < 2:
            logging.warning(f"Not enough groups to perform test for ELO category '{elo}'.")
            continue
        perform_statistical_test('EvalChange', subset, test_results, test_type='independent_t')

def analyze_time_in_position_categories(df, test_results):
    """
    Analyzes in what kinds of positions ADHD players take longer to make moves.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing all moves.
    - test_results (list): List to store statistical test results.
    """
    # Define position categories
    position_categories = df['PositionComplexity'].unique()
    
    # Ensure 'TimeSpent' is numeric
    df['TimeSpent'] = pd.to_numeric(df['TimeSpent'], errors='coerce')

    # Drop rows with NaNs in 'TimeSpent'
    df = df.dropna(subset=['TimeSpent'])

    # Calculate average time spent per position category and group
    avg_time = df.groupby(['Group', 'PositionComplexity'])['TimeSpent'].mean().reset_index()

    # Plotting - Bar Plot
    plt.figure(figsize=(12,8))
    sns.barplot(x='PositionComplexity', y='TimeSpent', hue='Group', data=avg_time, palette='Set2')
    plt.title('Average Time Spent per Position Complexity')
    plt.ylabel('Average Time Spent (seconds)')
    plt.xlabel('Position Complexity')
    plt.legend(title='Group')
    plt.show()
    
    # Statistical Testing: Compare time spent between groups within each position category
    for category in position_categories:
        subset = df[df['PositionComplexity'] == category]
        if subset['Group'].nunique() < 2:
            logging.warning(f"Not enough groups to perform test for Position Complexity '{category}'.")
            continue
        perform_statistical_test('TimeSpent', subset, test_results, test_type='independent_t')

def analyze_performance_under_move_conditions(df, test_results):
    """
    Analyzes performance (EvalChange) under different move conditions for both groups.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing all moves.
    - test_results (list): List to store statistical test results.
    """
    # Define move condition categories
    move_conditions = df['MoveCondition'].unique()
    
    # Ensure 'EvalChange' is numeric
    df['EvalChange'] = pd.to_numeric(df['EvalChange'], errors='coerce')

    # Drop rows with NaNs in 'EvalChange'
    df = df.dropna(subset=['EvalChange'])

    # Calculate average EvalChange per move condition and group
    avg_perf = df.groupby(['Group', 'MoveCondition'])['EvalChange'].mean().reset_index()
    
    # Plotting - Bar Plot
    plt.figure(figsize=(12,8))
    sns.barplot(x='MoveCondition', y='EvalChange', hue='Group', data=avg_perf, palette='Set2')
    plt.title('Average Evaluation Change under Move Conditions')
    plt.ylabel('Average Evaluation Change (centipawns)')
    plt.xlabel('Move Condition')
    plt.legend(title='Group')
    plt.xticks(rotation=45)
    plt.show()
    
    # Statistical Testing: Compare EvalChange between groups within each move condition
    for condition in move_conditions:
        subset = df[df['MoveCondition'] == condition]
        if subset['Group'].nunique() < 2:
            logging.warning(f"Not enough groups to perform test for Move Condition '{condition}'.")
            continue
        perform_statistical_test('EvalChange', subset, test_results, test_type='independent_t')

# ----------------------- Main Execution -----------------------

def main():
    # ----------------------- 1. Fetch and Process ADHD Players' Games -----------------------
    
    adhd_games = []
    for username in ADHD_USERNAMES:
        logging.info(f"Fetching games for user '{username}'...")
        user_games = fetch_lichess_games(username, max_games=50)  # Adjust max_games as needed
        adhd_games.extend(user_games)
    
    if not adhd_games:
        logging.warning("No ADHD games fetched. Exiting analysis.")
    
    # Initialize the chess engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        logging.info(f"Initialized Stockfish engine at '{STOCKFISH_PATH}'.")
    except FileNotFoundError:
        logging.critical(f"Stockfish executable not found at '{STOCKFISH_PATH}'. Please update the path.")
        return
    except Exception as e:
        logging.critical(f"Failed to initialize Stockfish engine: {e}")
        return

    # ----------------------- 2. Process ADHD Players' Games -----------------------
    
    if adhd_games:
        logging.info("Processing ADHD players' games...")
        adhd_moves_df = process_games(adhd_games, group_label='ADHD', engine=engine)
    else:
        logging.warning("No ADHD games to process.")
        adhd_moves_df = pd.DataFrame()

    # ----------------------- 3. Fetch and Process General Population Games -----------------------
    
    logging.info("Fetching general population games...")
    general_games = process_pgn_file(GENERAL_PGN_FILE_PATH, max_games=15000)  # Adjust max_games as needed
    
    if general_games:
        logging.info("Processing general population games...")
        general_moves_df = process_games(general_games, group_label='General', engine=engine)
    else:
        logging.warning("No General population games to process.")
        general_moves_df = pd.DataFrame()

    # ----------------------- 4. Combine Datasets -----------------------
    
    logging.info("Combining datasets...")
    all_moves_df = pd.concat([adhd_moves_df, general_moves_df], ignore_index=True)

    # ----------------------- 5. Data Cleaning -----------------------
    
    logging.info("Cleaning data...")
    # Drop moves without necessary data
    required_columns = ['TimeSpent', 'Evaluation', 'EvalChange', 'WhiteElo', 'BlackElo']
    all_moves_df = all_moves_df.dropna(subset=required_columns)

    # Ensure 'IsSacrifice' is boolean
    all_moves_df['IsSacrifice'] = all_moves_df['IsSacrifice'].fillna(False).astype(bool)

    # Convert relevant columns to numeric types
    numeric_columns = ['TimeSpent', 'Evaluation', 'EvalChange', 'WhiteElo', 'BlackElo']
    for col in numeric_columns:
        all_moves_df[col] = pd.to_numeric(all_moves_df[col], errors='coerce')

    # Drop rows with NaNs resulted from non-numeric conversion
    all_moves_df = all_moves_df.dropna(subset=numeric_columns)

    # Optional: Print data types to verify
    logging.info("\nData Types After Conversion:")
    logging.info(all_moves_df.dtypes)

    logging.info("\nSample Data:")
    logging.info(all_moves_df.head())

    # ----------------------- 6. Statistical Testing -----------------------
    
    logging.info("\nPerforming statistical tests...")
    
    # Define a list to store test results
    test_results = []

    # ----------------------- 7. Analysis and Plotting -----------------------
    
    logging.info("Generating plots and performing statistical tests...")
    plot_performance_under_time_pressure(all_moves_df, test_results)
    plot_accuracy_vs_time(all_moves_df, test_results)
    plot_error_rate(all_moves_df, test_results)
    plot_time_management(all_moves_df, test_results)
    stratify_by_elo(all_moves_df, test_results)
    
    # New Analysis Functions
    analyze_time_in_position_categories(all_moves_df, test_results)
    analyze_performance_under_move_conditions(all_moves_df, test_results)
    
    # ----------------------- 8. Display Statistical Test Results -----------------------
    
    logging.info("\n----------------------- Statistical Test Results -----------------------\n")
    results_df = pd.DataFrame(test_results)
    
    if not results_df.empty:
        # Apply Bonferroni correction for multiple comparisons
        num_tests = len(results_df)
        results_df['Adjusted p-value'] = results_df['p-value'] * num_tests
        results_df['Adjusted p-value'] = results_df['Adjusted p-value'].apply(lambda x: min(x, 1.0))
        
        # Determine significance after correction
        results_df['Significant'] = results_df['Adjusted p-value'] < 0.05
        
        # Display the results
        print(results_df[['Variable', 'Test', 'Statistic', 'p-value', 'Adjusted p-value', 'Significant']])
        
        logging.info("\nNote: p-values have been adjusted using the Bonferroni correction for multiple comparisons.\n")
    else:
        logging.info("No statistical tests were performed.")
    
    # ----------------------- Save Results to CSV (Optional) -----------------------
    
    # Uncomment the following lines to save the test results to a CSV file
    # try:
    #     results_df.to_csv('statistical_test_results.csv', index=False)
    #     logging.info("Statistical test results saved to 'statistical_test_results.csv'.")
    # except Exception as e:
    #     logging.error(f"Failed to save statistical test results: {e}")
    
    # ----------------------- Cleanup -----------------------
    
    # Close the chess engine
    engine.quit()
    
    logging.info("Analysis complete.")

# ----------------------- Execute the Script -----------------------
if __name__ == "__main__":
    main()
# %%
    
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate error counts and percentages
error_counts = all_moves_df.groupby(['Group', 'ErrorCategory']).size().reset_index(name='Count')
total_counts = all_moves_df.groupby('Group')['ErrorCategory'].count().reset_index(name='Total')
error_counts = error_counts.merge(total_counts, on='Group')
error_counts['Percentage'] = (error_counts['Count'] / error_counts['Total']) * 100

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='ErrorCategory', y='Percentage', hue='Group', data=error_counts, palette='Set3')
plt.title('Error Rate Comparison')
plt.ylabel('Percentage of Moves (%)')
plt.xlabel('Error Category')
plt.legend(title='Group')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Group', y='TimeSpent', data=all_moves_df, palette='Set2')
plt.title('Time Spent per Move by Group')
plt.ylabel('Time Spent on Move (seconds)')
plt.xlabel('Group')
plt.show()


plt.figure(figsize=(10,6))
sns.scatterplot(x='TimeSpent', y='EvalChange', hue='Group', data=all_moves_df, alpha=0.3, palette='Set1')
sns.lineplot(x='TimeSpent', y='EvalChange', hue='Group', data=all_moves_df, ci=None, estimator='mean', palette='Set1')
plt.title('Move Accuracy Relative to Move Time')
plt.xlabel('Time Spent on Move (seconds)')
plt.ylabel('Evaluation Change (centipawns)')
plt.legend(title='Group')
plt.show()


