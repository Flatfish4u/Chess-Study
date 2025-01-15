# Load required libraries
library(tidyverse)
library(ggplot2)
library(patchwork)
library(lme4)
library(parameters)
library(scales)

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
