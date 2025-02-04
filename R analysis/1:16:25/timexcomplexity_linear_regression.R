# Required libraries
library(ggplot2)
library(gridExtra)
library(dplyr)
library(knitr)

# Analysis function
analyze_complexity_time <- function(dataset, elo_group, tc_group) {
  # Subset data
  subset_data <- subset(combined_processed_data, 
                        elo_bracket == elo_group & 
                          time_control_category == tc_group)
  
  # Run model
  model <- lm(time_spent ~ position_complexity * player_type, 
              data = subset_data)
  
  # Get correlations
  adhd_data <- subset_data[subset_data$player_type == "ADHD", ]
  nonadhd_data <- subset_data[subset_data$player_type == "Non-ADHD", ]
  
  adhd_cor <- cor.test(adhd_data$time_spent, adhd_data$position_complexity)
  nonadhd_cor <- cor.test(nonadhd_data$time_spent, nonadhd_data$position_complexity)
  
  # Return results
  list(
    model = summary(model),
    adhd_cor = adhd_cor,
    nonadhd_cor = nonadhd_cor,
    n_adhd = nrow(adhd_data),
    n_nonadhd = nrow(nonadhd_data)
  )
}

# Plotting function
make_complexity_plot <- function(dataset, elo_group, tc_group) {
  subset_data <- subset(dataset, 
                        elo_bracket == elo_group & 
                          time_control_category == tc_group)
  
  ggplot(subset_data, 
         aes(x = position_complexity, 
             y = time_spent, 
             color = player_type)) +
    geom_smooth(method = "lm", se = TRUE) +
    geom_point(alpha = 0.1) +
    scale_color_manual(values = c("ADHD" = "red", "Non-ADHD" = "blue")) +
    labs(title = paste("Time vs Complexity:", elo_group, "-", tc_group),
         x = "Position Complexity",
         y = "Time Spent (seconds)") +
    theme_minimal() +
    facet_wrap(~game_phase) +
    coord_cartesian(ylim = c(0, 60))
}

# Results table function
create_summary_table <- function(dataset, elo_group, tc_group) {
  subset_data <- subset(dataset, 
                        elo_bracket == elo_group & 
                          time_control_category == tc_group)
  
  model <- lm(time_spent ~ position_complexity * player_type, 
              data = subset_data)
  model_summary <- summary(model)
  
  data.frame(
    elo_bracket = elo_group,
    time_control = tc_group,
    n_adhd = sum(subset_data$player_type == "ADHD"),
    n_nonadhd = sum(subset_data$player_type == "Non-ADHD"),
    intercept = model_summary$coefficients[1,1],
    intercept_p = model_summary$coefficients[1,4],
    complexity_effect = model_summary$coefficients[2,1],
    complexity_p = model_summary$coefficients[2,4],
    group_effect = model_summary$coefficients[3,1],
    group_p = model_summary$coefficients[3,4],
    interaction_effect = model_summary$coefficients[4,1],
    interaction_p = model_summary$coefficients[4,4],
    r_squared = model_summary$r.squared
  )
}

# Run analysis
elo_groups <- levels(combined_processed_data$elo_bracket)
tc_groups <- c("Bullet", "Blitz", "Rapid")

# Initialize storage
results_list <- list()
plots_list <- list()
results_rows <- list()

# Main analysis loop
for(elo in elo_groups) {
  for(tc in tc_groups) {
    key <- paste(elo, tc)
    
    # Store results
    results_list[[key]] <- analyze_complexity_time(combined_processed_data, elo, tc)
    plots_list[[key]] <- make_complexity_plot(combined_processed_data, elo, tc)
    results_rows[[key]] <- create_summary_table(combined_processed_data, elo, tc)
    
    # Print current results
    cat("\n====", key, "====\n")
    print(results_list[[key]]$model$coefficients)
    cat("\nSample sizes:\n")
    cat("ADHD:", results_list[[key]]$n_adhd, "\n")
    cat("Non-ADHD:", results_list[[key]]$n_nonadhd, "\n")
  }
}

# Combine all results into one table
results_table <- do.call(rbind, results_rows)

# Round numeric columns
numeric_cols <- sapply(results_table, is.numeric)
results_table[numeric_cols] <- round(results_table[numeric_cols], 4)

# Save results
write.csv(results_table, "complexity_time_analysis.csv", row.names = FALSE)

# Display plots
for(i in seq(1, length(plots_list), 4)) {
  grid.arrange(grobs = plots_list[i:min(i+3, length(plots_list))], ncol = 2)
}

# Display formatted table
kable(results_table)


