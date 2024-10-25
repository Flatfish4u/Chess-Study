#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:44:56 2024

@author: benjaminrosales
"""



# Import necessary libraries
import requests
import pandas as pd
import chess.pgn
import io
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import chess.engine

# Function to fetch all games from Lichess API in PGN format with pagination, limiting to max_games
def fetch_all_lichess_games(username, max_games=1000):
    games_per_request = 100  # Max number of games per request
    all_games = []
    offset = 0

    while len(all_games) < max_games:
        url = f'https://lichess.org/api/games/user/{username}'
        params = {
            'max': games_per_request,
            'clocks': True,    # Include clock times for moves
            'evals': False,    # Exclude server evaluations
            'pgnInJson': False, # Fetch PGN directly
            'opening': False,
            'moves': True,     # Include moves
            'offset': offset    # Paginate results
        }

        response = requests.get(url, params=params, stream=True)

        if response.status_code == 200:
            pgn_content = response.content.decode('utf-8').strip()
            games = pgn_content.split("\n\n\n")  # Split by game separator

            if not games or len(games[0].strip()) == 0:
                break

            all_games.extend(games)

            # Update the offset for the next batch
            offset += games_per_request

            # Limit to the max number of games
            if len(all_games) >= max_games:
                all_games = all_games[:max_games]
                break
        else:
            print(f"Error fetching games for {username}: {response.status_code}")
            break

    return all_games

# Function to parse Lichess games from PGN format with clocks
def parse_games_with_clocks(pgn_data):
    games = []
    for game_pgn in pgn_data:
        game = chess.pgn.read_game(io.StringIO(game_pgn))
        if game:
            games.append(game)
    return games

# Function to extract game data and moves with clock times
def extract_game_data_with_clocks(game):
    moves = []
    times = []
    board = game.board()
    node = game

    while node.variations:
        next_node = node.variations[0]
        move = next_node.move
        san = board.san(move)

        # Extract clock time from comments
        comment = next_node.comment
        time_match = re.search(r'\[%clk (\d+):(\d{2}):(\d{2})\]', comment)
        if time_match:
            hours, minutes, seconds = map(int, time_match.groups())
            time_seconds = hours * 3600 + minutes * 60 + seconds
        else:
            time_match = re.search(r'\[%clk (\d{1,2}):(\d{2})\]', comment)
            if time_match:
                minutes, seconds = map(int, time_match.groups())
                time_seconds = minutes * 60 + seconds
            else:
                time_seconds = None  # Time not available

        moves.append(san)
        times.append(time_seconds)
        board.push(move)
        node = next_node

    game_data = {
        'white_player': game.headers['White'],
        'black_player': game.headers['Black'],
        'result': game.headers['Result'],
        'moves': moves,
        'times': times,
    }
    return game_data

# Function to evaluate moves using Stockfish
def evaluate_moves(game, time_limit=0.1):
    # Replace the path with the one obtained from 'which stockfish'
    engine = chess.engine.SimpleEngine.popen_uci('/opt/homebrew/bin/stockfish')
    board = game.board()
    evaluations = []

    for move in game.mainline_moves():
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = info['score'].white().score(mate_score=100000)
        evaluations.append(score)

    engine.quit()
    return evaluations


# Function to analyze game data
def analyze_game(game_data, evaluations):
    moves = game_data['moves']
    times = game_data['times']
    analysis = []

    for i in range(len(moves)):
        move_number = i + 1
        time_remaining = times[i] if times[i] is not None else None
        eval_score = evaluations[i] if i < len(evaluations) else None

        if i > 0:
            if times[i - 1] is not None and times[i] is not None:
                time_spent = times[i - 1] - times[i]
            else:
                time_spent = None
            if evaluations[i - 1] is not None and evaluations[i] is not None:
                eval_change = evaluations[i] - evaluations[i - 1]
            else:
                eval_change = None
        else:
            time_spent = None
            eval_change = None

        under_time_pressure = time_remaining is not None and time_remaining < 20

        move_data = {
            'move_number': move_number,
            'move': moves[i],
            'time_remaining': time_remaining,
            'time_spent': time_spent,
            'evaluation': eval_score,
            'eval_change': eval_change,
            'under_time_pressure': under_time_pressure,
            'player': 'White' if (i % 2 == 0) else 'Black',
        }
        analysis.append(move_data)

    return analysis

# Function to plot Time Spent vs Evaluation Change
def plot_time_vs_accuracy(analysis_df, username):
    # Remove None values
    df = analysis_df.dropna(subset=['time_spent', 'eval_change'])
    if df.empty:
        print(f"No data to plot for {username}")
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(df['time_spent'], df['eval_change'], alpha=0.5)
    plt.title(f"Time Spent vs. Evaluation Change for {username}")
    plt.xlabel("Time Spent on Move (seconds)")
    plt.ylabel("Change in Evaluation (centipawns)")
    plt.grid(True)
    plt.show()

# Function to plot Performance Under Time Pressure
def plot_performance_under_time_pressure(analysis_df, username):
    # Filter moves under time pressure
    pressure_moves = analysis_df[analysis_df['under_time_pressure'] == True]
    non_pressure_moves = analysis_df[analysis_df['under_time_pressure'] == False]

    # Remove None values
    pressure_moves = pressure_moves['eval_change'].dropna()
    non_pressure_moves = non_pressure_moves['eval_change'].dropna()

    if pressure_moves.empty or non_pressure_moves.empty:
        print(f"Not enough data to plot time pressure performance for {username}")
        return

    # Boxplot of evaluation changes
    plt.figure(figsize=(10, 6))
    data = [pressure_moves, non_pressure_moves]
    plt.boxplot(data, labels=['Under Time Pressure', 'Not Under Time Pressure'])
    plt.title(f"Evaluation Change Under Time Pressure for {username}")
    plt.ylabel("Change in Evaluation (centipawns)")
    plt.grid(True)
    plt.show()

# Function to categorize errors
def categorize_errors(analysis_df):
    conditions = [
        (analysis_df['eval_change'] >= 200),
        (analysis_df['eval_change'] >= 100) & (analysis_df['eval_change'] < 200),
        (analysis_df['eval_change'] >= 50) & (analysis_df['eval_change'] < 100),
        (analysis_df['eval_change'] <= -200),
        (analysis_df['eval_change'] <= -100) & (analysis_df['eval_change'] > -200),
        (analysis_df['eval_change'] <= -50) & (analysis_df['eval_change'] > -100),
    ]
    choices = ['Blunder (Loss)', 'Mistake (Loss)', 'Inaccuracy (Loss)',
               'Blunder (Gain)', 'Mistake (Gain)', 'Inaccuracy (Gain)']
    analysis_df['error_category'] = np.select(conditions, choices, default='Good Move')
    return analysis_df

# Function to plot Error Categories
def plot_error_categories(analysis_df, username):
    error_counts = analysis_df['error_category'].value_counts()
    error_counts.plot(kind='bar')
    plt.title(f"Error Categories for {username}")
    plt.xlabel("Error Category")
    plt.ylabel("Number of Moves")
    plt.show()

# List of usernames
usernames = ['teoeo', 'Tobermorey', 'apostatlet', 'LovePump1000', 'Stuntmanandy', 
             'Banfy_B', 'ChessyChesterton12', 'Yastoon', 'Timy1976', 'SonnyDayz11', 'xiroir']

# For each user, perform analysis
for username in usernames:
    print(f"Processing games for {username}...")
    # Fetch games
    pgn_data = fetch_all_lichess_games(username, max_games=10)  # Adjust max_games as needed
    games = parse_games_with_clocks(pgn_data)

    all_analysis = []
    for game in games:
        game_data = extract_game_data_with_clocks(game)
        if len(game_data['moves']) == 0:
            continue  # Skip games without moves
        try:
            evaluations = evaluate_moves(game)
        except Exception as e:
            print(f"Error evaluating moves for {username}: {e}")
            continue
        game_analysis = analyze_game(game_data, evaluations)
        all_analysis.extend(game_analysis)

    # Create DataFrame
    analysis_df = pd.DataFrame(all_analysis)
    if analysis_df.empty:
        print(f"No analysis data for {username}")
        continue

    # Categorize errors
    analysis_df = categorize_errors(analysis_df)

    # Save analysis to CSV (optional)
    # analysis_df.to_csv(f'{username}_analysis.csv', index=False)

    # Plot graphs
    plot_time_vs_accuracy(analysis_df, username)
    plot_performance_under_time_pressure(analysis_df, username)
    plot_error_categories(analysis_df, username)

    # Additional analysis (e.g., time management patterns)
    # You can add more plots or statistical analysis as needed
    
import matplotlib.colors as mcolors

# Function to plot Time Spent vs Evaluation Change with symmetric log scale
def plot_time_vs_accuracy(analysis_df, username):
    # Remove None values and zeros (to avoid issues with log scale)
    df = analysis_df.dropna(subset=['time_spent', 'eval_change'])
    if df.empty:
        print(f"No data to plot for {username}")
        return

    plt.figure(figsize=(10, 6))

    # Create a scatter plot
    plt.scatter(df['time_spent'], df['eval_change'], alpha=0.5)

    # Use a symmetric logarithmic scale
    plt.yscale('symlog', linthresh=50)

    plt.title(f"Time Spent vs. Evaluation Change for {username}")
    plt.xlabel("Time Spent on Move (seconds)")
    plt.ylabel("Change in Evaluation (centipawns)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# Function to plot Time Spent vs Evaluation Change with capped eval_change
def plot_time_vs_accuracy(analysis_df, username):
    # Remove None values
    df = analysis_df.dropna(subset=['time_spent', 'eval_change'])
    if df.empty:
        print(f"No data to plot for {username}")
        return

    # Cap eval_change values
    cap_value = 500  # Set the cap value as needed
    df['eval_change_capped'] = df['eval_change'].clip(lower=-cap_value, upper=cap_value)

    plt.figure(figsize=(10, 6))
    plt.scatter(df['time_spent'], df['eval_change_capped'], alpha=0.5)
    plt.title(f"Time Spent vs. Evaluation Change for {username}")
    plt.xlabel("Time Spent on Move (seconds)")
    plt.ylabel(f"Change in Evaluation (capped at ±{cap_value} centipawns)")
    plt.grid(True)
    plt.show()

# Function to plot Time Spent vs Absolute Evaluation Change
def plot_time_vs_accuracy_abs(analysis_df, username):
    # Remove None values
    df = analysis_df.dropna(subset=['time_spent', 'eval_change'])
    if df.empty:
        print(f"No data to plot for {username}")
        return

    # Calculate absolute value of eval_change
    df['abs_eval_change'] = df['eval_change'].abs()

    plt.figure(figsize=(10, 6))
    plt.scatter(df['time_spent'], df['abs_eval_change'], alpha=0.5)
    plt.title(f"Time Spent vs. Absolute Evaluation Change for {username}")
    plt.xlabel("Time Spent on Move (seconds)")
    plt.ylabel("Absolute Change in Evaluation (centipawns)")
    plt.grid(True)
    plt.show()

# Function to plot Time Spent vs Evaluation Change using Hexbin
def plot_time_vs_accuracy_hexbin(analysis_df, username):
    # Remove None values
    df = analysis_df.dropna(subset=['time_spent', 'eval_change'])
    if df.empty:
        print(f"No data to plot for {username}")
        return

    plt.figure(figsize=(10, 6))
    plt.hexbin(df['time_spent'], df['eval_change'], gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(label='Count')
    plt.title(f"Time Spent vs. Evaluation Change for {username}")
    plt.xlabel("Time Spent on Move (seconds)")
    plt.ylabel("Change in Evaluation (centipawns)")
    plt.grid(True)
    plt.show()

def plot_time_vs_accuracy(analysis_df, username, method='symlog', cap_value=500):
    # Remove None values
    df = analysis_df.dropna(subset=['time_spent', 'eval_change'])
    if df.empty:
        print(f"No data to plot for {username}")
        return

    plt.figure(figsize=(10, 6))

    if method == 'symlog':
        plt.scatter(df['time_spent'], df['eval_change'], alpha=0.5)
        plt.yscale('symlog', linthresh=50)
        plt.ylabel("Change in Evaluation (centipawns, symlog scale)")
    elif method == 'capped':
        df['eval_change_capped'] = df['eval_change'].clip(lower=-cap_value, upper=cap_value)
        plt.scatter(df['time_spent'], df['eval_change_capped'], alpha=0.5)
        plt.ylabel(f"Change in Evaluation (capped at ±{cap_value} centipawns)")
    elif method == 'absolute':
        df['abs_eval_change'] = df['eval_change'].abs()
        plt.scatter(df['time_spent'], df['abs_eval_change'], alpha=0.5)
        plt.ylabel("Absolute Change in Evaluation (centipawns)")
    elif method == 'hexbin':
        plt.hexbin(df['time_spent'], df['eval_change'], gridsize=50, cmap='viridis', mincnt=1)
        plt.colorbar(label='Count')
        plt.ylabel("Change in Evaluation (centipawns)")
    else:
        plt.scatter(df['time_spent'], df['eval_change'], alpha=0.5)
        plt.ylabel("Change in Evaluation (centipawns)")

    plt.title(f"Time Spent vs. Evaluation Change for {username}")
    plt.xlabel("Time Spent on Move (seconds)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# Plot using symmetric log scale
plot_time_vs_accuracy(analysis_df, username, method='symlog')

# Plot with capped eval_change
plot_time_vs_accuracy(analysis_df, username, method='capped', cap_value=500)

# Plot using absolute values
plot_time_vs_accuracy(analysis_df, username, method='absolute')

# Plot using hexbin
plot_time_vs_accuracy(analysis_df, username, method='hexbin')



print("Analysis complete.")


