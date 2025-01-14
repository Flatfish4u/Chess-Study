# Load libraries
library(tidyverse)

# Define ELO brackets and labels
elo_breaks <- c(-Inf, 800, 1200, 1600, 2000, Inf)
elo_labels <- c("≤800", "801-1200", "1201-1600", "1601-2000", "2000+")

# Preprocess ADHD player data
adhd_player_moves <- adhd_moves_df %>%
  filter(is_adhd_move == TRUE, event != "Rated correspondence game") %>%
  mutate(
    time_spent = replace_na(time_spent, 0),
    eval_change = replace_na(eval_change, 0),
    winning_chances_change = replace_na(winning_chances_change, 0),
    avg_game_elo = (white_elo + black_elo) / 2,
    elo_bracket = cut(avg_game_elo, breaks = elo_breaks, labels = elo_labels, include.lowest = TRUE)
  )

# Preprocess general population data
general_moves_df <- general_moves_df %>%
  filter(event != "Rated Correspondence game") %>%
  mutate(
    time_spent = replace_na(time_spent, 0),
    eval_change = replace_na(eval_change, 0),
    winning_chances_change = replace_na(winning_chances_change, 0),
    avg_game_elo = (white_elo + black_elo) / 2,
    elo_bracket = cut(avg_game_elo, breaks = elo_breaks, labels = elo_labels, include.lowest = TRUE)
  )

# Compute ADHD sample sizes per ELO bracket
elo_bracket_counts <- adhd_player_moves %>%
  group_by(elo_bracket) %>%
  summarise(n_adhd = n(), .groups = "drop") %>%
  mutate(n_adhd = pmax(n_adhd, 1)) # Ensure no zero counts for sampling

# Filter and sample general moves
set.seed(432)
stratified_general_games <- general_moves_df %>%
  filter(
    white_elo >= min(adhd_player_moves$white_elo),
    white_elo <= max(adhd_player_moves$white_elo),
    black_elo >= min(adhd_player_moves$black_elo),
    black_elo <= max(adhd_player_moves$black_elo)
  ) %>%
  left_join(elo_bracket_counts, by = "elo_bracket") %>%
  group_by(elo_bracket) %>%
  group_modify(~ slice_sample(.x, n = .x$n_adhd[1])) %>%
  ungroup()

# Combine ADHD and sampled general moves data
combined_processed_data <- bind_rows(
  adhd_player_moves %>% mutate(player_type = "ADHD"),
  stratified_general_games %>% mutate(player_type = "Non-ADHD")
)

# Summarize data
stats_summary <- combined_processed_data %>%
  group_by(elo_bracket, player_type, time_control_category) %>%
  summarise(
    n_moves = n(),
    mean_quality = mean(abs(eval_change), na.rm = TRUE),
    sd_quality = sd(abs(eval_change), na.rm = TRUE),
    mean_time = mean(time_spent, na.rm = TRUE),
    sd_time = sd(time_spent, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n_moves >= 100)

# Output results for inspection
print(stats_summary)

