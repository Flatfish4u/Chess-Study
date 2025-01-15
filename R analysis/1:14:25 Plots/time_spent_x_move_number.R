library(tidyverse)
library(ggplot2)

# Define y-axis limits for each game type
y_limits <- list(
  "Bullet" = 120,  # ~1 minute
  "Blitz" = 180,   # ~3 minutes
  "Rapid" = 600    # ~10 minutes
)

# Set your desired path for saving the PDF
output_path <- "/Users/benjaminrosales/Downloads/time_usage_plots.pdf"

# Open a PDF device
pdf(file = output_path, width = 8, height = 6)

# Filter valid game types
valid_game_types <- c("Bullet", "Blitz", "Rapid")
filtered_data <- combined_processed_data %>%
  filter(time_control_category %in% valid_game_types)

# Iterate through game types and ELO brackets
for (game_type in unique(filtered_data$time_control_category)) {
  y_limit <- y_limits[[game_type]]
  
  game_data <- filtered_data %>%
    filter(time_control_category == game_type)
  
  for (elo_bracket in unique(game_data$elo_bracket)) {
    elo_data <- game_data %>%
      filter(elo_bracket == elo_bracket)
    
    # Time Spent vs Complexity
    p1 <- ggplot(elo_data, aes(x = position_complexity, y = time_spent, color = player_type)) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "lm", se = TRUE) +
      ylim(0, y_limit) +
      labs(
        title = paste(game_type, "-", elo_bracket, ": Time Spent vs Complexity"),
        x = "Position Complexity",
        y = "Time Spent (seconds)",
        color = "Player Type"
      ) +
      theme_minimal()
    
    # Time Spent vs Move Number
    p2 <- ggplot(elo_data, aes(x = move_number, y = time_spent, color = player_type)) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "lm", se = TRUE) +
      ylim(0, y_limit) +
      labs(
        title = paste(game_type, "-", elo_bracket, ": Time Spent vs Move Number"),
        x = "Move Number",
        y = "Time Spent (seconds)",
        color = "Player Type"
      ) +
      theme_minimal()
    
    # Print plots to the PDF
    print(p1)
    print(p2)
  }
}

# Close the PDF device
dev.off()

cat("All plots have been saved to:", output_path, "\n")
