library(tidyverse)
library(tidymodels)
library(arrow)
library(yardstick)

# Test-set evaluation of all five fitted models, producing the performance
# comparison table used in the thesis. AUC-ROC is reported once per model;
# Precision@k, Lift@k and NDCG@k are computed at k = 1%, 5% and 10%. Everything
# runs on db_test, the held-out set locked after the split in Modeling_simple.R.


# 0. Dependencies ----

source("../2. Feature Engineering/Feature_Engineering.R")
source("../3. Data Modeling/Metrics.R")


# 1. Load models and test data ----

cat("Loading fitted models...\n")

fit_logistic      <- readRDS("../3. Data Modeling/outputs/fit_logistic.rds")
fit_elastic_net   <- readRDS("../3. Data Modeling/outputs/fit_elastic_net.rds")
fit_decision_tree <- readRDS("../3. Data Modeling/outputs/fit_decision_tree.rds")
fit_random_forest <- readRDS("../3. Data Modeling/outputs/fit_random_forest.rds")
fit_xgboost       <- readRDS("../3. Data Modeling/outputs/fit_xgboost.rds")

cat("Loading test data...\n")

split    <- readRDS("../3. Data Modeling/outputs/split.rds")
db_test  <- testing(split)

# unique_id guarantees one row per transaction
stopifnot(n_distinct(db_test$unique_id) == nrow(db_test))
cat("Test rows verified unique via unique_id\n")

cat("Test rows:", nrow(db_test),
    "| Flagged:", sum(db_test$risk_score == "1"),
    "(", round(100 * mean(db_test$risk_score == "1"), 2), "%)\n")


# 2. Predictions on the test set ----

# .pred_1 is the risk score (probability of the suspicious class), joined back
# to unique_id, index_nr and the true label. db_test still carries the
# case_weights column from training; predict() ignores it.
cat("\nGenerating predictions on test set...\n")

predict_risk <- function(fit, model_name, test_data) {
  cat("  Predicting:", model_name, "...\n")
  preds <- predict(fit, new_data = test_data, type = "prob") |>
    bind_cols(
      test_data |> select(unique_id, index_nr, risk_score)
    ) |>
    mutate(model = model_name) |>
    select(model, unique_id, index_nr, risk_score, risk_score_predicted = .pred_1)
  preds
}

predictions_all <- bind_rows(
  predict_risk(fit_logistic,      "Logistic Regression", db_test),
  predict_risk(fit_elastic_net,   "Elastic Net",         db_test),
  predict_risk(fit_decision_tree, "Decision Tree",       db_test),
  predict_risk(fit_random_forest, "Random Forest",       db_test),
  predict_risk(fit_xgboost,       "XGBoost",             db_test)
)

cat("Predictions complete.\n")
cat("Total rows:", nrow(predictions_all), "\n")


# 3. Evaluation metrics ----

# One row per model x k. AUC-ROC is threshold-independent and repeated across
# the k rows for readability.
cat("\nComputing evaluation metrics...\n")

k_values <- c(0.01, 0.05, 0.10)

model_names <- unique(predictions_all$model)

compute_model_metrics <- function(model_name, preds_df, k_values) {

  preds <- preds_df |> filter(model == model_name)

  truth    <- as.numeric(preds$risk_score) - 1
  estimate <- preds$risk_score_predicted

  auc_val <- roc_auc_vec(
    truth    = preds$risk_score,
    estimate = estimate,
    event_level = "second"
  )

  map_dfr(k_values, function(k) {
    tibble(
      model          = model_name,
      k_pct          = k * 100,
      auc_roc        = round(auc_val, 4),
      precision_at_k = round(compute_precision_at_k(truth, estimate, k), 4),
      lift_at_k      = round(compute_lift_at_k(truth, estimate, k),      4),
      ndcg_at_k      = round(compute_ndcg_at_k(truth, estimate, k),      4)
    )
  })
}

evaluation_results <- map_dfr(model_names, function(m) {
  compute_model_metrics(m, predictions_all, k_values)
})

model_order <- c("Logistic Regression", "Elastic Net", "Decision Tree",
                 "Random Forest", "XGBoost")

evaluation_results <- evaluation_results |>
  mutate(model = factor(model, levels = model_order)) |>
  arrange(model, k_pct)

cat("\n--- Test Set Evaluation Results ---\n")
print(evaluation_results, n = Inf)


# 3.1 Summary comparison table ----

# Wide format (one row per model) as used in the thesis comparison table.
evaluation_table <- evaluation_results |>
  pivot_wider(
    id_cols     = model,
    names_from  = k_pct,
    names_glue  = "{.value}_k{k_pct}",
    values_from = c(precision_at_k, lift_at_k, ndcg_at_k)
  ) |>
  left_join(
    evaluation_results |>
      distinct(model, auc_roc),
    by = "model"
  ) |>
  select(model, auc_roc,
         precision_at_k_k1, lift_at_k_k1, ndcg_at_k_k1,
         precision_at_k_k5, lift_at_k_k5, ndcg_at_k_k5,
         precision_at_k_k10, lift_at_k_k10, ndcg_at_k_k10)

cat("\n--- Summary Comparison Table ---\n")
print(evaluation_table, n = Inf)


# 4. Save outputs ----

if (!dir.exists("outputs")) dir.create("outputs")

saveRDS(predictions_all,    "outputs/predictions_all.rds")
saveRDS(evaluation_results, "outputs/evaluation_results.rds")
write_csv(evaluation_table, "outputs/evaluation_table.csv")

cat("\nEvaluation.R complete — outputs written to 4. Evaluation/outputs/\n")
