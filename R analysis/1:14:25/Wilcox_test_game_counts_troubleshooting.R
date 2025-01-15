# === Step 8: Statistical Testing ===
cat("\n=== Statistical Testing: Wilcoxon Rank-Sum Test ===\n")

# Loop through each ELO bracket and time control category
results_list <- list()  # Initialize a list to store results

# Get unique combinations of ELO bracket and time control
elo_time_combinations <- combined_processed_data %>%
  distinct(elo_bracket, time_control_category) %>%
  arrange(elo_bracket, time_control_category)

for (i in seq_len(nrow(elo_time_combinations))) {
  # Extract current combination
  current_elo_bracket <- elo_time_combinations$elo_bracket[i]
  current_time_control <- elo_time_combinations$time_control_category[i]
  
  cat("\n=== Testing for ELO Bracket:", current_elo_bracket, 
      "and Time Control:", current_time_control, "===\n")
  
  # Subset data for the current group
  group_data <- combined_processed_data %>%
    filter(
      elo_bracket == current_elo_bracket, 
      time_control_category == current_time_control
    )
  
  # Check the player type distribution
  player_type_dist <- table(group_data$player_type)
  cat("Player type distribution:\n")
  print(player_type_dist)
  
  # If both player types are present, run the Wilcoxon test
  if (length(unique(group_data$player_type)) == 2) {
    cat("Running Wilcoxon test...\n")
    
    # Perform the test
    wilcox_test <- tryCatch(
      wilcox.test(time_spent ~ player_type, data = group_data, exact = FALSE),
      error = function(e) {
        cat("Wilcoxon test failed:\n")
        print(e)
        return(NULL)
      }
    )
    
    # Save the results if the test was successful
    if (!is.null(wilcox_test)) {
      cat("Wilcoxon test successful! P-value:", wilcox_test$p.value, "\n")
      results_list[[paste(current_elo_bracket, current_time_control, sep = "_")]] <- list(
        elo_bracket = current_elo_bracket,
        time_control = current_time_control,
        p_value = wilcox_test$p.value,
        test_statistic = wilcox_test$statistic,
        mean_adhd = mean(group_data$time_spent[group_data$player_type == "ADHD"], na.rm = TRUE),
        mean_non_adhd = mean(group_data$time_spent[group_data$player_type == "Non-ADHD"], na.rm = TRUE),
        n_adhd = sum(group_data$player_type == "ADHD"),
        n_non_adhd = sum(group_data$player_type == "Non-ADHD")
      )
    }
  } else {
    cat("Skipping group due to insufficient data or only one player type.\n")
  }
}

# Combine results into a data frame for easy viewing
test_results <- do.call(rbind, lapply(results_list, as.data.frame))

cat("\n=== Final Wilcoxon Test Results ===\n")
print(test_results)

# Save results to a CSV file for inspection
write_csv(test_results, "wilcoxon_test_results.csv")
cat("Results saved to 'wilcoxon_test_results.csv'\n")

# Visualize mean time spent by player type
library(ggplot2)

ggplot(combined_processed_data, aes(x = player_type, y = time_spent, fill = player_type)) +
  geom_boxplot() +
  facet_grid(elo_bracket ~ time_control_category) +
  theme_minimal() +
  labs(
    title = "Time Spent by Player Type Across ELO Brackets and Time Controls",
    x = "Player Type",
    y = "Time Spent (s)"
  )


# ANOVA: Does time_spent differ by elo_bracket within ADHD players?
anova_results <- aov(time_spent ~ elo_bracket + player_type, data = combined_processed_data)
summary(anova_results)

# Kruskal-Wallis for non-normal distributions
kruskal_results <- combined_processed_data %>%
  group_by(player_type) %>%
  summarise(
    p_value = kruskal.test(time_spent ~ elo_bracket, data = cur_data())$p.value
  )

print(kruskal_results)


# Post-hoc test to compare differences between ELO brackets
tukey_results <- TukeyHSD(aov(time_spent ~ elo_bracket, data = combined_processed_data))
print(tukey_results)

# Visualize Tukey HSD results
plot(tukey_results, las = 1, col = "blue")


# Include interaction in ANOVA
interaction_anova <- aov(time_spent ~ elo_bracket * player_type, data = combined_processed_data)
summary(interaction_anova)



