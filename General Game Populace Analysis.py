#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on [Date]

@author: benjaminrosales
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

# ----------------------- Configuration -----------------------

# Configure plotting style
sns.set(style='whitegrid')

# Replace with the actual path to your general population PGN file
GENERAL_PGN_FILE_PATH = '/Users/benjaminrosales/Desktop/Chess Study/Comparison Games/lichess_db_standard_rated_2017-05.pgn'  # **Update this path**

# ----------------------- Helper Functions -----------------------

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
    time_match = re.search(r'\[%clk (\d+):(\d{2}):(\d{2})\]', comment)
    if time_match:
        hours, minutes, seconds = map(int, time_match.groups())
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    else:
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
    if eval_change <= -200:
        return 'Blunder'
    elif eval_change <= -100:
        return 'Mistake'
    elif eval_change <= -50:
        return 'Inaccuracy'
    else:
        return 'Good Move'

def process_games(games, group_label):
    """
    Processes a list of chess games and extracts move-level data.
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
            }

            all_moves.append(move_data)

            prev_time[board.turn] = time_remaining
            prev_eval = eval_score

            board.push(move)
            node = next_node

    return pd.DataFrame(all_moves)

def fetch_lichess_games(username, max_games=2000):
    """
    Fetches games from Lichess API for a given username.
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
    except requests.exceptions.RequestException as e:
        print(f"Request exception for {username}: {e}")
        return []

    games = []

    if response.status_code == 200:
        pgn_content = response.content.decode('utf-8').strip()
        pgn_io = io.StringIO(pgn_content)
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            games.append(game)
    else:
        print(f"Error fetching games for {username}: {response.status_code}")

    return games

def process_pgn_file(pgn_file_path, max_games=None):
    """
    Processes a PGN file and extracts games up to max_games.
    """
    games = []
    with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file:
        game_counter = 0
        pbar = tqdm(total=max_games if max_games else None, desc='Processing general population games')
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
    return games

# ----------------------- Main Execution -----------------------

def main():
    # ----------------------- 1. Fetch and Process ADHD Players' Games -----------------------
    
    # List of ADHD players' usernames
    adhd_usernames = ['teoeo', 'Tobermorey', 'apostatlet', 'LovePump1000', 'Stuntmanandy', 
                      'Banfy_B', 'ChessyChesterton12', 'Yastoon', 'Timy1976', 'SonnyDayz11', 'xiroir']
    
    adhd_games = []
    for username in adhd_usernames:
        print(f"Fetching games for {username}...")
        user_games = fetch_lichess_games(username, max_games=1500)  # Adjust max_games as needed
        adhd_games.extend(user_games)
    
    print("Processing ADHD players' games...")
    adhd_moves_df = process_games(adhd_games, group_label='ADHD')
    
    # ----------------------- 2. Fetch and Process General Population Games -----------------------
    
    print("Fetching general population games...")
    general_games = process_pgn_file(GENERAL_PGN_FILE_PATH, max_games=15000)  # Adjust max_games as needed
    
    print("Processing general population games...")
    general_moves_df = process_games(general_games, group_label='General')
    
    # ----------------------- 3. Combine Datasets -----------------------
    
    print("Combining datasets...")
    all_moves_df = pd.concat([adhd_moves_df, general_moves_df], ignore_index=True)
    
    # ----------------------- 4. Data Cleaning -----------------------
    
    print("Cleaning data...")
    # Drop moves without necessary data
    all_moves_df = all_moves_df.dropna(subset=['TimeSpent', 'Evaluation', 'EvalChange', 'WhiteElo', 'BlackElo'])
    
    # ----------------------- 5. Statistical Testing -----------------------
    
    print("Performing statistical tests...")
    
    # Define a list to store test results
    test_results = []
    
    # Function to perform and store statistical tests
    def perform_statistical_test(var, data, test_type='independent_t'):
        """
        Performs a statistical test between ADHD and General groups for a given variable.
        """
        group1 = data[data['Group'] == 'ADHD'][var]
        group2 = data[data['Group'] == 'General'][var]
        
        # Check normality
        sample_size = min(500, len(group1), len(group2))  # Shapiro-Wilk has a max sample size
        if sample_size < 3:
            print(f"Not enough data to perform test for variable '{var}'.")
            return
        
        stat1, p1 = stats.shapiro(group1.sample(sample_size))
        stat2, p2 = stats.shapiro(group2.sample(sample_size))
        normal1 = p1 > 0.05
        normal2 = p2 > 0.05
        
        # Check variance equality
        stat_levene, p_levene = stats.levene(group1, group2)
        equal_var = p_levene > 0.05
        
        if normal1 and normal2:
            # Use Independent Samples t-test
            stat, p = stats.ttest_ind(group1, group2, equal_var=equal_var)
            test_used = 'Independent t-test'
        else:
            # Use Mann-Whitney U test
            stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_used = 'Mann-Whitney U test'
        
        # Store results
        test_results.append({
            'Variable': var,
            'Test': test_used,
            'Statistic': stat,
            'p-value': p
        })
    
    # Function to perform chi-squared test for categorical variables
    def perform_chi_squared_test(category_var, data):
        """
        Performs Chi-Squared test between ADHD and General groups for a given categorical variable.
        """
        contingency_table = pd.crosstab(data['Group'], data[category_var])
        
        # Perform Chi-Squared test
        stat, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Store results
        test_results.append({
            'Variable': category_var,
            'Test': 'Chi-Squared test',
            'Statistic': stat,
            'p-value': p
        })
    
    # ----------------------- 6. Analysis and Plotting -----------------------
    
    # Function to plot performance under time pressure
    def plot_performance_under_time_pressure(df):
        """
        Plots average evaluation change under time pressure for both groups.
        Also performs statistical testing.
        """
        # Only consider moves under time pressure
        df_pressure = df[df['UnderTimePressure']]
        
        if df_pressure.empty:
            print("No moves under time pressure to plot.")
            return
        
        # Calculate average evaluation change under time pressure for each group
        avg_eval_change = df_pressure.groupby('Group')['EvalChange'].mean().reset_index()
        
        # Plotting - Bar Plot
        plt.figure(figsize=(8,6))
        ax = sns.barplot(x='Group', y='EvalChange', data=avg_eval_change, palette='Set2')
        plt.title('Average Evaluation Change Under Time Pressure (<20s)')
        plt.ylabel('Average Evaluation Change (centipawns)')
        plt.xlabel('Group')
        
        # Perform statistical test
        perform_statistical_test('EvalChange', df_pressure, test_type='independent_t')
        
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
    
    # Function to plot move accuracy relative to move time
    def plot_accuracy_vs_time(df):
        """
        Plots move accuracy relative to move time, comparing both groups.
        Also performs statistical testing.
        """
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
        perform_statistical_test('EvalChange', df, test_type='independent_t')
    
    # Function to plot error rate comparison
    def plot_error_rate(df):
        """
        Plots error rate comparison between both groups.
        Also performs Chi-Squared testing.
        """
        # Calculate error counts
        error_counts = df.groupby(['Group', 'ErrorCategory']).size().reset_index(name='Count')
        
        # Normalize counts to get percentages
        total_counts = df.groupby('Group')['ErrorCategory'].count().reset_index(name='Total')
        error_counts = error_counts.merge(total_counts, on='Group')
        error_counts['Percentage'] = (error_counts['Count'] / error_counts['Total']) * 100
        
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
        perform_chi_squared_test('ErrorCategory', df)
    
    # Function to plot time management patterns
    def plot_time_management(df):
        """
        Plots time management patterns for both groups.
        Also performs statistical testing.
        """
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
        perform_statistical_test('TimeSpent', df, test_type='independent_t')
    
    # Function to stratify by ELO and compare groups
    def stratify_by_elo(df):
        """
        Stratifies data by ELO categories and compares both groups within each category.
        Also performs statistical testing within each ELO category.
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
                print(f"Not enough groups to perform test for ELO category {elo}.")
                continue
            perform_statistical_test('EvalChange', subset, test_type='independent_t')
    
    # ----------------------- Execute Plotting Functions -----------------------
    
    print("Generating plots and performing statistical tests...")
    plot_performance_under_time_pressure(all_moves_df)
    plot_accuracy_vs_time(all_moves_df)
    plot_error_rate(all_moves_df)
    plot_time_management(all_moves_df)
    stratify_by_elo(all_moves_df)
    
    # ----------------------- 7. Display Statistical Test Results -----------------------
    
    print("\n----------------------- Statistical Test Results -----------------------\n")
    results_df = pd.DataFrame(test_results)
    
    # Apply Bonferroni correction for multiple comparisons
    num_tests = len(results_df)
    results_df['Adjusted p-value'] = results_df['p-value'] * num_tests
    results_df['Adjusted p-value'] = results_df['Adjusted p-value'].apply(lambda x: min(x, 1.0))
    
    # Determine significance after correction
    results_df['Significant'] = results_df['Adjusted p-value'] < 0.05
    
    # Display the results
    print(results_df[['Variable', 'Test', 'Statistic', 'p-value', 'Adjusted p-value', 'Significant']])
    
    print("\nNote: p-values have been adjusted using the Bonferroni correction for multiple comparisons.\n")
    
    # ----------------------- Save Results to CSV (Optional) -----------------------
    
    # Uncomment the following lines to save the test results to a CSV file
    # results_df.to_csv('statistical_test_results.csv', index=False)
    
    print("Analysis complete.")

# ----------------------- Run the Script -----------------------
if __name__ == "__main__":
    main()










