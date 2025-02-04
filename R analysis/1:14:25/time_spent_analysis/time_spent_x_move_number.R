# Load required libraries
library(tidyverse)
library(ggplot2)
library(patchwork)
library(lme4)
library(parameters)
library(scales)
library(effsize) # For effect size calculations

# Define custom chess theme
custom_theme <- theme_minimal() + 
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1, "lines"),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9)
  )

# Colors for ADHD and Non-ADHD
chess_colors <- c("ADHD" = "#FF6B6B", "Non-ADHD" = "#4ECDC4")

# Function to generate time analysis plots
create_time_analysis_plots <- function(combined_processed_data, elo_range, time_type) {
  filtered_data <- combined_processed_data %>%
    filter(elo_bracket == elo_range, time_control_category == time_type)
  
  # Adjust y-axis limits for Bullet games
  y_limit <- if (time_type == "Bullet") {
    10 # Limit to 10 seconds for better visualization
  } else {
    NA # No limit for other time controls
  }
  
  # Time spent vs. position complexity with linear regression and SD ribbon
  p1 <- ggplot(filtered_data, aes(x = position_complexity, y = time_spent, color = player_type, fill = player_type)) +
    geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = TRUE, alpha = 0.2) +
    labs(
      title = paste("Time Spent vs Position Complexity -", elo_range, time_type),
      x = "Position Complexity",
      y = "Time Spent (seconds)",
      color = "Player Type",
      fill = "Player Type"
    ) +
    scale_color_manual(values = chess_colors) +
    scale_fill_manual(values = chess_colors) +
    custom_theme +
    ylim(0, y_limit)
  
  # Time spent vs. move number with linear regression and SD ribbon
  p2 <- ggplot(filtered_data, aes(x = move_number, y = time_spent, color = player_type, fill = player_type)) +
    geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = TRUE, alpha = 0.2) +
    labs(
      title = paste("Time Spent vs Move Number -", elo_range, time_type),
      x = "Move Number",
      y = "Time Spent (seconds)",
      color = "Player Type",
      fill = "Player Type"
    ) +
    scale_color_manual(values = chess_colors) +
    scale_fill_manual(values = chess_colors) +
    custom_theme +
    ylim(0, y_limit)
  
  combined_plot <- (p1 / p2) + plot_annotation(
    title = paste("Time Analysis for", elo_range, time_type),
    theme = custom_theme
  )
  
  print(combined_plot) # Print plot directly to R console for visualization
  return(combined_plot)
}

# Statistical Analysis and Summary Table
create_statistical_summary <- function(combined_processed_data) {
  summary_stats <- combined_processed_data %>%
    group_by(player_type, elo_bracket, time_control_category) %>%
    summarise(
      mean_time_spent = mean(time_spent, na.rm = TRUE),
      sd_time_spent = sd(time_spent, na.rm = TRUE),
      mean_position_complexity = mean(position_complexity, na.rm = TRUE),
      sd_position_complexity = sd(position_complexity, na.rm = TRUE),
      mean_move_number = mean(move_number, na.rm = TRUE),
      sd_move_number = sd(move_number, na.rm = TRUE),
      .groups = "drop"
    )
  
  write.csv(summary_stats, "statistical_summary.csv", row.names = FALSE)
  return(summary_stats)
}

# Statistical Testing for Significant Differences
perform_statistical_tests <- function(combined_processed_data) {
  test_results <- combined_processed_data %>%
    group_by(elo_bracket, time_control_category) %>%
    summarise(
      p_value_time_spent = tryCatch(
        t.test(time_spent ~ player_type, data = .)$p.value,
        error = function(e) NA
      ),
      effect_size_time_spent = tryCatch(
        cohen.d(time_spent ~ player_type, data = .)$estimate,
        error = function(e) NA
      ),
      p_value_position_complexity = tryCatch(
        t.test(position_complexity ~ player_type, data = .)$p.value,
        error = function(e) NA
      ),
      effect_size_position_complexity = tryCatch(
        cohen.d(position_complexity ~ player_type, data = .)$estimate,
        error = function(e) NA
      ),
      p_value_move_number = tryCatch(
        t.test(move_number ~ player_type, data = .)$p.value,
        error = function(e) NA
      ),
      effect_size_move_number = tryCatch(
        cohen.d(move_number ~ player_type, data = .)$estimate,
        error = function(e) NA
      ),
      .groups = "drop"
    )
  
  write.csv(test_results, "statistical_test_results.csv", row.names = FALSE)
  return(test_results)
}

# Iterate over ELO brackets and time controls
for (elo in unique(combined_processed_data$elo_bracket)) {
  for (time_type in unique(combined_processed_data$time_control_category)) {
    # Skip classical games
    if (time_type == "Classical") {
      next
    }
    
    data_subset <- combined_processed_data %>%
      filter(elo_bracket == elo, time_control_category == time_type)
    
    if (nrow(data_subset) >= 100) {  # Minimum sample size
      plot <- create_time_analysis_plots(combined_processed_data, elo, time_type)
      ggsave(
        filename = paste0("plots/time_analysis_", elo, "_", time_type, ".png"),
        plot = plot,
        width = 12,
        height = 8,
        dpi = 300
      )
    }
  }
}

# Generate Statistical Summary
statistical_summary <- create_statistical_summary(combined_processed_data)
print(statistical_summary)

# Perform Statistical Tests
statistical_test_results <- perform_statistical_tests(combined_processed_data)
print(statistical_test_results)

# Mixed-effects models for statistical analysis
mixed_effects_results <- list()
for (time_type in unique(combined_processed_data$time_control_category)) {
  # Skip classical games
  if (time_type == "Classical") {
    next
  }
  
  model_data <- combined_processed_data %>%
    filter(time_control_category == time_type)
  
  mixed_effects_results[[time_type]] <- lmer(
    time_spent ~ player_type + position_complexity + move_number + (1 | game_id),
    data = model_data
  )
}

# Save model summaries
model_summaries <- lapply(mixed_effects_results, function(model) {
  parameters(model) %>% as.data.frame()
})

for (time_type in names(model_summaries)) {
  write.csv(model_summaries[[time_type]], paste0("model_summaries_", time_type, ".csv"), row.names = FALSE)
}


