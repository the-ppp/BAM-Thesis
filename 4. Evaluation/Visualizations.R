# Risk score distribution per model. Shows how predicted scores spread out and
# where the interpretability review sample falls within that spread: if a model
# piles most transactions near 0, the random "middle 20" draw lands near 0 too
# rather than near the centre of the axis. Run from the "4. Evaluation" directory.

library(tidyverse)

predictions_all <- readRDS("outputs/predictions_all.rds")

model_order <- c("Logistic Regression", "Elastic Net", "Decision Tree",
                 "Random Forest", "XGBoost")

predictions_all <- predictions_all |>
  mutate(model = factor(model, levels = model_order))

# ------------------------------------------------------------------
# 1. Numeric summary per model
# ------------------------------------------------------------------
summary_tbl <- predictions_all |>
  group_by(model) |>
  summarise(
    n        = n(),
    min      = min(risk_score_predicted),
    p10      = quantile(risk_score_predicted, 0.10),
    q1       = quantile(risk_score_predicted, 0.25),
    median   = median(risk_score_predicted),
    mean     = mean(risk_score_predicted),
    q3       = quantile(risk_score_predicted, 0.75),
    p90      = quantile(risk_score_predicted, 0.90),
    p99      = quantile(risk_score_predicted, 0.99),
    max      = max(risk_score_predicted),
    pct_below_0_05 = mean(risk_score_predicted < 0.05),
    .groups  = "drop"
  )

cat("\n--- Risk score distribution summary per model ---\n")
print(summary_tbl, width = Inf)

# ------------------------------------------------------------------
# 2. Histogram per model (full test set)
# ------------------------------------------------------------------
p_hist <- ggplot(predictions_all,
                 aes(risk_score_predicted)) +
  geom_histogram(bins = 50, fill = "#1F6FC0", colour = "white", linewidth = 0.1) +
  facet_wrap(~ model, ncol = 1, scales = "free_y") +
  labs(title = "Predicted risk score distribution (full test set)",
       x = "Predicted risk score", y = "Count") +
  theme_minimal(base_size = 11)

ggsave("outputs/risk_score_distribution_hist.png", p_hist,
       width = 9, height = 11, dpi = 130)

# ------------------------------------------------------------------
# 3. Log-scaled count histogram — reveals the tail when scores pile at 0
# ------------------------------------------------------------------
p_log <- ggplot(predictions_all,
                aes(risk_score_predicted)) +
  geom_histogram(bins = 50, fill = "#375623", colour = "white", linewidth = 0.1) +
  scale_y_log10() +
  facet_wrap(~ model, ncol = 1) +
  labs(title = "Predicted risk score distribution (log count scale)",
       subtitle = "Log y-axis exposes the sparse high-score tail",
       x = "Predicted risk score", y = "Count (log10)") +
  theme_minimal(base_size = 11)

ggsave("outputs/risk_score_distribution_log.png", p_log,
       width = 9, height = 11, dpi = 130)

# ------------------------------------------------------------------
# 4. ECDF per model — clearest view of where mass concentrates
# ------------------------------------------------------------------
p_ecdf <- ggplot(predictions_all,
                 aes(risk_score_predicted, colour = model)) +
  stat_ecdf(linewidth = 0.8) +
  labs(title = "Cumulative distribution of predicted risk scores",
       x = "Predicted risk score", y = "Cumulative proportion",
       colour = "Model") +
  theme_minimal(base_size = 11)

ggsave("outputs/risk_score_distribution_ecdf.png", p_ecdf,
       width = 9, height = 6, dpi = 130)

cat("\nSaved:\n",
    " outputs/risk_score_distribution_hist.png\n",
    " outputs/risk_score_distribution_log.png\n",
    " outputs/risk_score_distribution_ecdf.png\n")
