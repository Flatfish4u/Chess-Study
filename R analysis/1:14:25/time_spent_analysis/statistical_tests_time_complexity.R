##########################
# FULLY WORKING ANALYSIS
##########################

# Load required libraries
library(dplyr)
library(effsize) # For cohen.d()

#------------------------------------------------
# 0) Optional: Check if there's any leftover grouping
#------------------------------------------------
cat("Grouping variables before ungrouping:\n")
print(group_vars(combined_processed_data)) 
# If this is NOT character(0), you have leftover grouping that could cause confusion.

#------------------------------------------------
# 1) Ungroup to ensure everything starts cleanly
#------------------------------------------------
combined_processed_data <- combined_processed_data %>% ungroup()

#------------------------------------------------
# 2) Summarize subgroup sample sizes
#    (ADHD vs. Non-ADHD in each combination)
#------------------------------------------------
group_sizes_per_row <- combined_processed_data %>%
  group_by(elo_bracket, time_control_category, player_type) %>%
  summarise(n = n(), .groups = "drop") %>%
  arrange(elo_bracket, time_control_category, player_type)

cat("\n===== SUBGROUP SAMPLE SIZES =====\n")
print(group_sizes_per_row)

#------------------------------------------------
# 3) Summarize means & SDs per subgroup
#   (time_spent, position_complexity, move_number)
#------------------------------------------------
group_means <- combined_processed_data %>%
  group_by(elo_bracket, time_control_category, player_type) %>%
  summarise(
    mean_time_spent = mean(time_spent, na.rm = TRUE),
    sd_time_spent   = sd(time_spent, na.rm = TRUE),
    mean_complexity = mean(position_complexity, na.rm = TRUE),
    sd_complexity   = sd(position_complexity, na.rm = TRUE),
    mean_move_num   = mean(move_number, na.rm = TRUE),
    sd_move_num     = sd(move_number, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  arrange(elo_bracket, time_control_category, player_type)

cat("\n===== SUBGROUP MEANS & SDs =====\n")
print(group_means)

#------------------------------------------------
# 4) OPTIONAL: Manually run a T-test on a single subset
#    to verify your group-wise approach. For instance,
#    the group (≤1000, Bullet) if it exists in your data.
#------------------------------------------------
cat("\n===== MANUAL T-TEST ON (≤1000, Bullet) =====\n")

manual_subset <- combined_processed_data %>%
  filter(elo_bracket == "≤1000", time_control_category == "Bullet")

cat("Count of ADHD vs Non-ADHD in this (≤1000, Bullet) subset:\n")
print(table(manual_subset$player_type))

manual_ttest <- t.test(time_spent ~ player_type, data = manual_subset)
manual_cohen <- cohen.d(time_spent ~ player_type, data = manual_subset)

cat("Manual subset (≤1000, Bullet) t-test p-value: ", manual_ttest$p.value, "\n")
cat("Manual subset (≤1000, Bullet) Cohen's d:     ", manual_cohen$estimate, "\n")

#------------------------------------------------
# 5) FULL GROUP-WISE STATISTICAL ANALYSIS
#   (time_spent, position_complexity, move_number)
#   with T-tests and Cohen's d
#------------------------------------------------
cat("\n===== FULL GROUP-WISE STATISTICAL ANALYSIS =====\n")

test_results_debug <- combined_processed_data %>%
  group_by(elo_bracket, time_control_category) %>%
  do({
    # Subset of data for this ELO bracket & time control
    subset_data <- .
    
    # -- T-test & Cohen’s d: time_spent --
    out_spent <- tryCatch(t.test(time_spent ~ player_type, data = subset_data),
                          error = function(e) NA)
    eff_spent <- tryCatch(cohen.d(time_spent ~ player_type, data = subset_data),
                          error = function(e) NA)
    
    # -- T-test & Cohen’s d: position_complexity --
    out_complex <- tryCatch(t.test(position_complexity ~ player_type, data = subset_data),
                            error = function(e) NA)
    eff_complex <- tryCatch(cohen.d(position_complexity ~ player_type, data = subset_data),
                            error = function(e) NA)
    
    # -- T-test & Cohen’s d: move_number --
    out_move <- tryCatch(t.test(move_number ~ player_type, data = subset_data),
                         error = function(e) NA)
    eff_move <- tryCatch(cohen.d(move_number ~ player_type, data = subset_data),
                         error = function(e) NA)
    
    tibble(
      p_value_time_spent               = if (is.list(out_spent)) out_spent$p.value else NA,
      effect_size_time_spent           = if (is.list(eff_spent)) eff_spent$estimate else NA,
      p_value_position_complexity      = if (is.list(out_complex)) out_complex$p.value else NA,
      effect_size_position_complexity  = if (is.list(eff_complex)) eff_complex$estimate else NA,
      p_value_move_number              = if (is.list(out_move)) out_move$p.value else NA,
      effect_size_move_number          = if (is.list(eff_move)) eff_move$estimate else NA
    )
  }) %>%
  ungroup()

cat("\n===== FINAL TEST RESULTS TABLE =====\n")
print(test_results_debug)

#------------------------------------------------
# 6) (OPTIONAL) Save results to CSV
#------------------------------------------------
write.csv(test_results_debug, "~/Downloads/test_results_debug.csv", row.names = TRUE)



