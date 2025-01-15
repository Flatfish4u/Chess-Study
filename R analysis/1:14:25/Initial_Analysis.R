# === Step 1: Setup and Libraries ===
# Load the required library
library(tidyverse)

# Define ELO brackets and labels
elo_breaks <- c(-Inf, 1000, 1400, 1800, Inf)
elo_labels <- c("≤1000", "1001-1400", "1401-1800", "1801+")

cat("\n=== Processing ADHD and General Games ===\n")

#processing & filters for adhd

adhd_player_moves <- adhd_moves_df %>%
  filter(is_adhd_move == TRUE, event != "Rated correspondence game") %>%
  mutate(
    time_spent = replace_na(time_spent, 0),
    eval_change = replace_na(eval_change, 0),
    winning_chances_change = replace_na(winning_chances_change, 0),
    avg_game_elo = (white_elo + black_elo) / 2,  # Average ELO for the game
    elo_bracket = cut(avg_game_elo, breaks = elo_breaks, labels = elo_labels, include.lowest = TRUE)
  )
print(head(adhd_player_moves))

#genpop filters

# Filter general moves and clean up missing values
general_moves_df <- general_moves_df %>%
  filter(event != "Rated Correspondence game") %>%
  mutate(
    time_spent = replace_na(time_spent, 0),
    eval_change = replace_na(eval_change, 0),
    winning_chances_change = replace_na(winning_chances_change, 0),
    avg_game_elo = (white_elo + black_elo) / 2,  # Average ELO for the game
    elo_bracket = cut(avg_game_elo, breaks = elo_breaks, labels = elo_labels, include.lowest = TRUE)
  )

print(head(general_moves_df))

# Count ADHD moves by ELO bracket
adhd_elo_counts <- adhd_player_moves %>%
  count(elo_bracket, name = "n_adhd")

# Count general moves by ELO bracket
general_elo_counts <- general_moves_df %>%
  count(elo_bracket, name = "n_general")

# Combine counts into one table and calculate sample sizes
elo_counts <- left_join(adhd_elo_counts, general_elo_counts, by = "elo_bracket") %>%
  replace_na(list(n_adhd = 0, n_general = 0)) %>%
  mutate(sample_size = pmin(n_adhd, n_general))

cat("\n=== ADHD vs General Moves by ELO Bracket ===\n")
print(elo_counts)


# Ensure 'sample_size' is scalar per group
set.seed(432)
stratified_general_games <- general_moves_df %>%
  left_join(elo_counts %>% select(elo_bracket, sample_size), by = "elo_bracket") %>%
  group_by(elo_bracket, time_control_category) %>%
  group_modify(~ {
    n <- unique(.x$sample_size)  # Get the unique sample size for this group
    if (length(n) != 1 || is.na(n) || n <= 0) {
      # If no valid sample size, return an empty tibble
      tibble()
    } else {
      # Otherwise, sample 'n' rows
      slice_sample(.x, n = n)
    }
  }) %>%
  ungroup()

# === Step 6: Combine ADHD and General Data ===
cat("\nCombining ADHD and sampled general games...\n")

combined_processed_data <- bind_rows(
  adhd_player_moves %>% 
    mutate(player_type = "ADHD"),
  stratified_general_games %>% mutate(player_type = "Non-ADHD")
)

# Let's see what we're working with
str(combined_processed_data)
table(combined_processed_data$group)
table(combined_processed_data$elo_bracket)
table(combined_processed_data$time_control_category)

