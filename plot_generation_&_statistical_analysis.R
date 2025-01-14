# Load required libraries
library(tidyverse)
library(ggplot2)
library(patchwork)
library(lme4)
library(parameters)
library(scales)

# Create paired plots function
create_paired_plots <- function(data, elo_range, time_type) {
  # Set y-axis limit based on time control type
  y_limit <- switch(time_type,
                   "Bullet" = 120,
                   "Blitz" = 180,
                   "Rapid" = 600)
  
  # Filter data for this combination
  filtered_data <- data %>%
    filter(elo_bracket == elo_range,
           time_control_category == time_type)
  
  # Time spent vs complexity plot
  p1 <- ggplot(filtered_data, 
               aes(x = position_complexity, 
                   y = time_spent, 
                   color = player_type)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", level = 0.99) +
    ylim(0, y_limit) +
    scale_color_manual(values = chess_colors) +
    theme_chess +
    labs(
      title = paste(elo_range, time_type, "- Time vs Complexity"),
      x = "Position Complexity",
      y = "Time Spent (seconds)",
      color = "Player Type"
    )
  
  # Time usage plot
  p2 <- ggplot(filtered_data, 
               aes(x = move_number, 
                   y = (time_spent/initial_time_seconds)*100, 
                   color = player_type)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", level = 0.99) +
    ylim(0, 100) +
    scale_color_manual(values = chess_colors) +
    theme_chess +
    labs(
      title = paste(elo_range, time_type, "- Time Budget Usage"),
      x = "Move Number",
      y = "Percentage of Time Budget Used",
      color = "Player Type"
    )
  
  # Evaluation change plot
  p3 <- ggplot(filtered_data, 
               aes(x = player_type, y = eval_change, fill = player_type)) +
    geom_violin(trim = FALSE) +
    geom_boxplot(width = 0.1, fill = "white", color = "black") +
    scale_fill_manual(values = chess_colors) +
    theme_chess +
    labs(
      title = paste(elo_range, time_type, "- Evaluation Change"),
      x = "Player Type",
      y = "Evaluation Change (centipawns)",
      fill = "Player Type"
    )
  
  # Statistical tests
  time_test <- wilcox.test(time_spent ~ player_type, data = filtered_data)
  eval_test <- wilcox.test(eval_change ~ player_type, data = filtered_data)
  
  # Add statistical annotations
  stats_text <- sprintf(
    "Time Test p: %.3f\nEval Test p: %.3f",
    time_test$p.value,
    eval_test$p.value
  )
  
  # Combine plots with statistical annotation
  combined_plot <- (p1 + p2) / p3 +
    plot_annotation(
      title = paste("Analysis for", elo_range, "-", time_type),
      subtitle = stats_text,
      theme = theme_chess
    )
  
  return(combined_plot)
}

# Generate plots
plot_list <- list()

for(elo in unique(combined_processed_data$elo_bracket)) {
  for(time_type in time_controls) {
    # Check if we have enough data for this combination
    data_subset <- combined_processed_data %>%
      filter(elo_bracket == elo, 
             time_control_category == time_type)
    
    if(nrow(data_subset) >= 100) {  # Minimum sample size threshold
      plot_name <- paste(elo, time_type, sep="_")
      plot_list[[plot_name]] <- create_paired_plots(
        combined_processed_data, 
        elo, 
        time_type
      )
      
      # Save individual plots
      ggsave(
        filename = paste0("plots/", plot_name, ".png"),
        plot = plot_list[[plot_name]],
        width = 12,
        height = 8,
        dpi = 300
      )
    }
  }
}

# Statistical Analysis Section
# Create summary statistics for each combination
stats_summary <- combined_processed_data %>%
  group_by(elo_bracket, player_type, time_control_category) %>%
  summarise(
    n_moves = n(),
    mean_time = mean(time_spent, na.rm = TRUE),
    sd_time = sd(time_spent, na.rm = TRUE),
    mean_eval_change = mean(abs(eval_change), na.rm = TRUE),
    sd_eval_change = sd(abs(eval_change), na.rm = TRUE),
    mean_complexity = mean(position_complexity, na.rm = TRUE),
    sd_complexity = sd(position_complexity, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n_moves >= 100)  # Ensure sufficient sample size

# Mixed Effects Models
mixed_models <- list()

# Loop through each time control type
for(time_type in time_controls) {
  # Filter data
  model_data <- combined_processed_data %>%
    filter(time_control_category == time_type)
  
  # Time spent model
  mixed_models[[paste0(time_type, "_time")]] <- lmer(
    time_spent ~ player_type + position_complexity + (1|game_id),
    data = model_data
  )
  
  # Eval change model
  mixed_models[[paste0(time_type, "_eval")]] <- lmer(
    abs(eval_change) ~ player_type + position_complexity + (1|game_id),
    data = model_data
  )
}

# Extract and format model results
model_summaries <- lapply(mixed_models, function(model) {
  parameters(model) %>%
    as.data.frame()
})


print(plot_list)
