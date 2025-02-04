# Using dplyr for cleaner syntax
library(dplyr)

# Group and summarize
elo_time_summary <- combined_processed_data %>%
  group_by(elo_bracket, time_control_category, group) %>%
  summarise(
    n_moves = n(),
    mean_eval_change = mean(eval_change, na.rm = TRUE),
    sd_eval_change = sd(eval_change, na.rm = TRUE),
    mean_time_spent = mean(time_spent, na.rm = TRUE),
    sd_time_spent = sd(time_spent, na.rm = TRUE),
    mean_complexity = mean(position_complexity, na.rm = TRUE),
    sd_complexity = sd(position_complexity, na.rm = TRUE)
  ) %>%
  ungroup()

# View the results
print(elo_time_summary)

# For a specific ELO bracket and time control:
analyze_subset <- function(data, elo_level, time_control) {
  subset_data <- data %>%
    filter(
      elo_bracket == elo_level,
      time_control_category == time_control
    )
  
  # Wilcoxon test (non-parametric alternative to t-test)
  wilcox_result <- wilcox.test(
    eval_change ~ group,
    data = subset_data
  )
  
  # Summary statistics
  summary_stats <- subset_data %>%
    group_by(group) %>%
    summarise(
      n = n(),
      median_eval = median(eval_change, na.rm = TRUE),
      mean_eval = mean(eval_change, na.rm = TRUE)
    )
  
  return(list(
    test = wilcox_result,
    summary = summary_stats
  ))
}

results <- analyze_subset(
  data = combined_processed_data,
  elo_level = "≤1000",
  time_control = "Blitz"
)







