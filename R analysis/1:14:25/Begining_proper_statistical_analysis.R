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










# First, let's check what groups we actually have in our data

unique_groups <- unique(combined_processed_data$group)
print("Unique groups in data:")
print(unique_groups)

# Let's check our filtered subset
subset_check <- combined_processed_data %>%
  filter(
    elo_bracket == "1401-1800",
    time_control_category == "Blitz"
  ) %>%
  group_by(group) %>%
  summarise(count = n())

print("\nDistribution in 1401-1800 Blitz games:")
print(subset_check)

# Modified analysis function with more robust checks
analyze_subset <- function(data, elo_level, time_control) {
  # First check the subset
  subset_data <- data %>%
    filter(
      elo_bracket == elo_level,
      time_control_category == time_control
    )
  
  # Check if we have data
  if(nrow(subset_data) == 0) {
    return("No data found for this combination")
  }
  
  # Print group counts for debugging
  group_counts <- table(subset_data$group)
  print("Group counts in subset:")
  print(group_counts)
  
  # Summary statistics first
  summary_stats <- subset_data %>%
    group_by(group) %>%
    summarise(
      n = n(),
      median_eval = median(eval_change, na.rm = TRUE),
      mean_eval = mean(eval_change, na.rm = TRUE),
      sd_eval = sd(eval_change, na.rm = TRUE)
    )
  
  # Only do Wilcoxon test if we have exactly 2 groups
  if(length(unique(subset_data$group)) == 2) {
    wilcox_result <- wilcox.test(
      eval_change ~ group,
      data = subset_data
    )
    return(list(
      summary = summary_stats,
      test = wilcox_result
    ))
  } else {
    return(list(
      summary = summary_stats,
      message = "Could not perform Wilcoxon test - need exactly 2 groups"
    ))
  }
}
