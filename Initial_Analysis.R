# Core setup
library(tidyverse)
library(ggplot2)
library(scales)
library(lme4)
library(parameters)

# Your original theme
theme_chess <- theme_minimal() + 
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1, "lines"),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9)
  )

chess_colors <- c("ADHD" = "#FF6B6B", "Non-ADHD" = "#4ECDC4")

### Data PreProcessing - KEEPING YOUR ORIGINAL CODE
adhd_player_moves <- adhd_moves_df %>%
  filter(is_adhd_move == TRUE) %>%
  filter(event != "Rated correspondence game")

general_moves_df <- general_moves_df %>%
  filter(event != "Rated Correspondence game")

# Handle NA values
adhd_player_moves$time_spent[is.na(adhd_player_moves$time_spent)] <- 0
general_moves_df$time_spent[is.na(general_moves_df$time_spent)] <- 0
adhd_player_moves$eval_change[is.na(adhd_player_moves$eval_change)] <- 0
general_moves_df$eval_change[is.na(general_moves_df$eval_change)] <- 0
adhd_player_moves$winning_chances_change[is.na(adhd_player_moves$winning_chances_change)] <- 0
general_moves_df$winning_chances_change[is.na(general_moves_df$winning_chances_change)] <- 0

# Remove adhd_player column if it exists
adhd_player_moves <- adhd_player_moves %>%
  select(-any_of("adhd_player"))
general_moves_df <- general_moves_df %>%
  select(-any_of("adhd_player"))

# granular ELO 
elo_breaks <- c(-Inf, 800, 1200, 1600, 2000, Inf)
elo_labels <- c("≤800", "801-1200", "1201-1600", "1601-2000", "2000+")

# Modify the sampling to be more precise
set.seed(432)
n_adhd_games <- nrow(adhd_player_moves)
stratified_general_games <- general_moves_df %>%
  # First filter to exact ADHD ELO range
  filter(
    white_elo >= min(adhd_player_moves$white_elo),
    white_elo <= max(adhd_player_moves$white_elo),
    black_elo >= min(adhd_player_moves$black_elo),
    black_elo <= max(adhd_player_moves$black_elo)
  ) %>%
  # Create more precise ELO brackets
  mutate(
    avg_game_elo = (white_elo + black_elo) / 2,
    elo_bracket = cut(avg_game_elo, 
                      breaks = elo_breaks,
                      include.lowest = TRUE),
    # Create weight based on distance from ADHD median
    sampling_weight = 1/(abs(avg_game_elo - median(adhd_player_moves$white_elo)) + 1)
  ) %>%
  # Sample proportionally to ADHD distribution
  group_by(elo_bracket) %>%
  slice_sample(
    n = ceiling(n_adhd_games/10),  # Sampling from 10 brackets now
    weight_by = sampling_weight
  ) %>%
  ungroup() %>%
  pull(game_id)

comparable_general_moves <- general_moves_df %>%
  filter(game_id %in% stratified_general_games)

combined_processed_data <- bind_rows(
  adhd_player_moves %>% 
    mutate(
      player_type = "ADHD",
      avg_game_elo = (white_elo + black_elo) / 2,
      elo_bracket = cut(avg_game_elo, 
                        breaks = elo_breaks,
                        labels = elo_labels,
                        include.lowest = TRUE)
    ),
  comparable_general_moves %>% 
    mutate(
      player_type = "Non-ADHD",
      avg_game_elo = (white_elo + black_elo) / 2,
      elo_bracket = cut(avg_game_elo, 
                        breaks = elo_breaks,
                        labels = elo_labels,
                        include.lowest = TRUE)
    )
)

rm(combined_processed_data$event)


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

combined_processed_data <- 
  combined_processed_data %>%
  filter(
    time_control_category != "Classical"
  )

###
time_controls <- c("Bullet", "Blitz", "Rapid")




