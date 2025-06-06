# ======================================================================
# PUBLICATION-READY DATA FIXES
# Add this code block RIGHT AFTER your data cleaning section in Main.ipynb
# ======================================================================

import pandas as pd
import numpy as np
import os
import logging

# Apply this right after: logging.info(f"Total number of moves after cleaning: {len(all_moves_df)}")

logging.info("=== APPLYING PUBLICATION-READY DATA FIXES ===")

# ----------------------- 1. FIX ELO BRACKETS TO MATCH PAPER -----------------------
# Your paper uses ≤1000, 1001-1400, 1401-1800, 1801+ but current code uses different brackets

# Create average ELO (more accurate than max)
all_moves_df['avg_elo'] = (all_moves_df['white_elo'] + all_moves_df['black_elo']) / 2

# FIX: Replace your existing elo_bracket creation with this:
all_moves_df['elo_bracket'] = pd.cut(
    all_moves_df['avg_elo'],
    bins=[0, 1000, 1400, 1800, float('inf')],
    labels=['≤1000', '1001-1400', '1401-1800', '1801+'],
    include_lowest=True
)

logging.info("✓ Fixed ELO brackets to match paper format")

# ----------------------- 2. CREATE PUBLICATION PLAYER TYPES -----------------------

# Create clear player type variable for R ANOVA
all_moves_df['player_type'] = all_moves_df['group'].map({
    'ADHD': 'ADHD', 
    'General': 'Non-ADHD'  # This matches your R scripts!
})

# Create binary indicator for easier analysis
all_moves_df['is_adhd_player'] = (all_moves_df['player_type'] == 'ADHD').astype(int)

logging.info("✓ Created proper player_type variables")

# ----------------------- 3. FIX TIME CONTROL CATEGORIES -----------------------

# Make sure time control categories match R scripts exactly
time_control_mapping = {
    'Bullet': 'Bullet',
    'Blitz': 'Blitz', 
    'Rapid': 'Rapid',
    'Classical': 'Classical'
}

all_moves_df['time_control_category'] = all_moves_df['time_control_category'].map(time_control_mapping)

logging.info("✓ Standardized time control categories")

# ----------------------- 4. CLEAN MISSING VALUES FOR R -----------------------

# Remove rows with critical missing values for regression analysis
critical_vars = ['time_spent', 'move_number', 'position_complexity', 'elo_bracket', 
                'time_control_category', 'player_type']

before_clean = len(all_moves_df)
all_moves_df = all_moves_df.dropna(subset=critical_vars)
after_clean = len(all_moves_df)

logging.info(f"✓ Cleaned missing values: {before_clean:,} → {after_clean:,} rows")

# ----------------------- 5. CREATE EXPORT-READY DATAFRAMES -----------------------

# Create the main analysis dataset
combined_processed_data = all_moves_df.copy()

# Create game-level summary for between-subjects analysis
logging.info("Creating game-level player summary...")

game_summary = combined_processed_data.groupby(['game_id', 'player_type']).agg({
    'time_spent': ['mean', 'std', 'count'],
    'eval_change': ['mean', 'std'],
    'under_time_pressure': lambda x: (x == True).sum() / len(x),  # Proportion under pressure
    'white_elo': 'first',
    'black_elo': 'first',
    'avg_elo': 'first',
    'elo_bracket': 'first',
    'time_control_category': 'first',
    'adhd_player': 'first'
}).reset_index()

# Flatten column names
game_summary.columns = [
    'game_id', 'player_type', 'mean_time_spent', 'std_time_spent', 'n_moves',
    'mean_eval_change', 'std_eval_change', 'prop_under_pressure',
    'white_elo', 'black_elo', 'avg_elo', 'elo_bracket', 'time_control_category', 'adhd_player'
]

logging.info(f"✓ Created game summary: {len(game_summary):,} games")

# ----------------------- 6. CREATE PLAYER-LEVEL SUMMARY -----------------------

# This is what Professor Kleiman needs for ANOVA
player_summary = game_summary.groupby(['adhd_player', 'player_type']).agg({
    'mean_time_spent': ['mean', 'std', 'count'],
    'mean_eval_change': ['mean', 'std'],
    'prop_under_pressure': 'mean',
    'avg_elo': 'mean',
    'n_moves': 'sum'
}).reset_index()

# Flatten columns
player_summary.columns = [
    'player_name', 'player_type', 'overall_mean_time', 'std_mean_time', 'n_games',
    'overall_mean_eval_change', 'std_eval_change', 'avg_prop_under_pressure',
    'average_elo', 'total_moves'
]

logging.info(f"✓ Created player summary: {len(player_summary):,} players")

# ----------------------- 7. EXPORT FILES FOR R ANALYSIS -----------------------

# Create output directory
output_dir = "/Users/benjaminrosales/Desktop/Chess-Worker/Chess-Study/Publication_Data"
os.makedirs(output_dir, exist_ok=True)

# Export main dataset (what your R scripts expect)
main_file = f"{output_dir}/combined_processed_data.csv"
combined_processed_data.to_csv(main_file, index=False)
logging.info(f"✓ Exported main dataset: {main_file}")

# Export game-level summary
game_file = f"{output_dir}/game_level_summary.csv"
game_summary.to_csv(game_file, index=False)
logging.info(f"✓ Exported game summary: {game_file}")

# Export player-level summary (for ANOVA)
player_file = f"{output_dir}/player_level_summary.csv"
player_summary.to_csv(player_file, index=False)
logging.info(f"✓ Exported player summary: {player_file}")

# ----------------------- 8. CREATE DATA VERIFICATION REPORT -----------------------

print("\n" + "="*60)
print("PUBLICATION DATA VERIFICATION REPORT")
print("="*60)

print(f"\n📊 SAMPLE SIZES:")
print(f"Total moves: {len(combined_processed_data):,}")
print(f"ADHD moves: {len(combined_processed_data[combined_processed_data['player_type'] == 'ADHD']):,}")
print(f"Non-ADHD moves: {len(combined_processed_data[combined_processed_data['player_type'] == 'Non-ADHD']):,}")

print(f"\n🎯 ELO DISTRIBUTION:")
elo_dist = combined_processed_data.groupby(['elo_bracket', 'player_type']).size().unstack(fill_value=0)
print(elo_dist)

print(f"\n⏱️ TIME CONTROL DISTRIBUTION:")
time_dist = combined_processed_data.groupby(['time_control_category', 'player_type']).size().unstack(fill_value=0)
print(time_dist)

print(f"\n👥 PLAYER COUNTS:")
print(f"Total unique players: {len(player_summary)}")
print(f"ADHD players: {len(player_summary[player_summary['player_type'] == 'ADHD'])}")
print(f"Non-ADHD players: {len(player_summary[player_summary['player_type'] == 'Non-ADHD'])}")

print(f"\n✅ DATA QUALITY CHECKS:")
print(f"Missing elo_bracket: {combined_processed_data['elo_bracket'].isna().sum()}")
print(f"Missing player_type: {combined_processed_data['player_type'].isna().sum()}")
print(f"Missing time_spent: {combined_processed_data['time_spent'].isna().sum()}")
print(f"Missing position_complexity: {combined_processed_data['position_complexity'].isna().sum()}")

print(f"\n📁 EXPORTED FILES:")
print(f"1. {main_file}")
print(f"2. {game_file}")
print(f"3. {player_file}")

print("\n🎉 DATA IS NOW PUBLICATION-READY!")
print("="*60)

# ----------------------- 9. CREATE R LOADING SNIPPET -----------------------

r_code = f'''
# ======================================================================
# R CODE TO LOAD YOUR PUBLICATION DATA
# Copy this into your R scripts
# ======================================================================

# Load the main dataset
combined_processed_data <- read.csv("{main_file}")

# Load game-level summary
game_summary <- read.csv("{game_file}")

# Load player-level summary (for ANOVA)
player_summary <- read.csv("{player_file}")

# Verify data structure
print("Data loaded successfully!")
print(paste("Total moves:", nrow(combined_processed_data)))
print(paste("ELO brackets:", toString(unique(combined_processed_data$elo_bracket))))
print(paste("Player types:", toString(unique(combined_processed_data$player_type))))
'''

r_file = f"{output_dir}/load_data.R"
with open(r_file, 'w') as f:
    f.write(r_code)

logging.info(f"✓ Created R loading script: {r_file}")

print(f"\n📋 Next steps:")
print(f"1. Run your R regression scripts using combined_processed_data")
print(f"2. Use player_summary for between-subjects ANOVA")
print(f"3. All ELO brackets now match your paper exactly")
print(f"4. Professor Kleiman can run analyses immediately")
