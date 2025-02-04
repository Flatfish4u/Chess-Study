library(tidyverse)

# Define your y-axis limit for the proportion
y_limits <- list(
  "Bullet" = 1,   # Cumulative proportion of time budget is normalized (0 to 1)
  "Blitz" = 1,
  "Rapid" = 1
)

# Filter valid game types and calculate cumulative time budget
filtered_data <- combined_processed_data %>%
  filter(time_control_category %in% c("Bullet", "Blitz", "Rapid")) %>%
  group_by(game_id, player_type, time_control_category, elo_bracket) %>%
  arrange(move_number) %>%  # Ensure moves are ordered
  mutate(
    cumulative_time_budget_used = cumsum(time_spent) / initial_time_seconds  # Cumulative proportion
  ) %>%
  ungroup()

# Aggregate the data: Calculate the average cumulative time usage at each move number
average_data <- filtered_data %>%
  group_by(player_type, time_control_category, elo_bracket, move_number) %>%
  summarise(
    avg_cumulative_time_budget = mean(cumulative_time_budget_used, na.rm = TRUE),
    se_cumulative_time_budget = sd(cumulative_time_budget_used, na.rm = TRUE) / sqrt(n()), # Standard error
    .groups = "drop"
  )

# Iterate through game types and ELO brackets
for (game_type in unique(average_data$time_control_category)) {
  y_limit <- y_limits[[game_type]]
  
  game_data <- average_data %>%
    filter(time_control_category == game_type)
  
  for (elo_bracket in unique(game_data$elo_bracket)) {
    elo_data <- game_data %>%
      filter(elo_bracket == elo_bracket)
    
    # Cumulative Time Budget Usage (Averages)
    p <- ggplot(elo_data, aes(x = move_number, y = avg_cumulative_time_budget, color = player_type)) +
      geom_line(size = 1) +  # Average trend line
      geom_ribbon(
        aes(ymin = avg_cumulative_time_budget - se_cumulative_time_budget,
            ymax = avg_cumulative_time_budget + se_cumulative_time_budget,
            fill = player_type),
        alpha = 0.2, color = NA
      ) +  # Confidence interval
      labs(
        title = paste("Cumulative Time Budget Usage (Average):", game_type, "-", elo_bracket),
        x = "Move Number",
        y = "Cumulative Proportion of Time Budget Used",
        color = "Player Type",
        fill = "Player Type"
      ) +
      scale_y_continuous(limits = c(0, y_limit)) +  # Limit y-axis to (0, 1)
      theme_minimal() +
      theme(
        plot.title = element_text(size = 14, face = "bold"),
        legend.position = "bottom"
      )
    
    # Save the plot
    ggsave(
      filename = paste0("/Users/benjaminrosales/Downloads/cumulative_time_budget_avg_", game_type, "_", elo_bracket, ".png"),
      plot = p,
      width = 8,
      height = 6,
      dpi = 300
    )
    
    # Print the plot for review
    print(p)
  }
}
