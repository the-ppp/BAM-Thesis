library(tidyverse)
library(tidymodels)
library(treeshap)

# Explanation layer for all five models: per-transaction feature contributions
# aggregated to the six risk categories, plus the interpretability review and a
# reproducibility check. Contributions are coefficient-based for the linear
# models, a variable-importance approximation for the decision tree, and
# TreeSHAP for the ensembles.


# 0. Dependencies and shared objects ----

source("../2. Feature Engineering/Feature_Engineering.R")
source("../3. Data Modeling/Metrics.R")

cat("Loading fitted models and data...\n")

fit_logistic      <- readRDS("../3. Data Modeling/outputs/fit_logistic.rds")
fit_elastic_net   <- readRDS("../3. Data Modeling/outputs/fit_elastic_net.rds")
fit_decision_tree <- readRDS("../3. Data Modeling/outputs/fit_decision_tree.rds")
fit_random_forest <- readRDS("../3. Data Modeling/outputs/fit_random_forest.rds")
fit_xgboost       <- readRDS("../3. Data Modeling/outputs/fit_xgboost.rds")

split           <- readRDS("../3. Data Modeling/outputs/split.rds")
db_test         <- testing(split)
stopifnot(n_distinct(db_test$unique_id) == nrow(db_test))
predictions_all <- readRDS("outputs/predictions_all.rds")

cat("Test rows:", nrow(db_test), "\n")


# 1. Feature-to-category mapping ----

# Each engineered feature is assigned to one of six audit risk categories
# (imposed from domain knowledge, not learned). Used to aggregate contributions
# into category scores and to rank features within each category. gl_type dummy
# columns are matched by prefix in get_category().
feature_category_map <- list(

  "Timing Behavior" = c(
    "is_weekend_post",
    "is_weekend_entry",
    "is_last_fiscal_day",
    "days_between_entry_and_posting",
    "user_period_entry_count"
  ),

  "Amount Behavior" = c(
    "abs_amount",
    "log_abs_amount",
    "is_round_1000",
    "journal_round_rate",
    "roundness_relative_to_journal"
  ),

  "User Behavior" = c(
    "user_account_rarity",
    "user_account_period_entry_count"
  ),

  "Journal Behavior" = c(
    "journal_account_match_rarity",
    "user_journal_rarity"
  ),

  "Text Flags" = c(
    "flag_correction_adjustment",
    "flag_accrual_unbilled",
    "flag_internal_transfer",
    "flag_bonus_payroll",
    "flag_credit_card",
    "flag_valuation",
    "flag_equity_shareholder",
    "flag_batch_processing"
  ),

  "Account Flags" = c(
    "revenue_account_flag",
    "accrual_account_flag",
    "provision_account_flag"
    # gl_type_* dummy columns assigned by prefix below
  )
)

categories <- names(feature_category_map)

# Map a feature name to its category: direct match, gl_type_ prefix, or _new
# suffix from step_novel() stripped and retried.
get_category <- function(feature_name, category_map) {
  for (cat in names(category_map)) {
    if (feature_name %in% category_map[[cat]]) return(cat)
  }
  if (startsWith(feature_name, "gl_type_")) return("Account Flags")
  base_name <- sub("_new$", "", feature_name)
  for (cat in names(category_map)) {
    if (base_name %in% category_map[[cat]]) return(cat)
  }
  return("Other")
}


# ---- Shared helper: assemble the explanation output ----
# Takes per-transaction feature contributions and returns one row per
# transaction with six category-contribution columns and six feature-ranking
# columns. All twelve columns are guaranteed present even when a category has no
# contributing features for a model.

build_explanation <- function(contributions, predictions, model_name) {

  contributions <- contributions |>
    mutate(category = map_chr(feature, get_category,
                              category_map = feature_category_map))

  # Category-level contributions: sum of feature contributions per category
  cat_contributions <- contributions |>
    group_by(unique_id, category) |>
    summarise(cat_contribution = sum(contribution, na.rm = TRUE),
              .groups = "drop") |>
    pivot_wider(
      id_cols     = unique_id,
      names_from  = category,
      values_from = cat_contribution,
      values_fill = 0
    )

  for (cat in categories) {
    if (!cat %in% names(cat_contributions)) {
      cat_contributions[[cat]] <- 0
    }
  }

  # Features per category, ranked by absolute contribution
  feature_rankings <- contributions |>
    group_by(unique_id, category) |>
    arrange(desc(abs(contribution)), .by_group = TRUE) |>
    summarise(
      ranked_features = paste(
        seq_along(feature),
        paste0(feature, " (", round(contribution, 4), ")"),
        sep = ". ",
        collapse = " | "
      ),
      .groups = "drop"
    ) |>
    pivot_wider(
      id_cols     = unique_id,
      names_from  = category,
      names_glue  = "{category}_features",
      values_from = ranked_features,
      values_fill = ""
    )

  for (cat in categories) {
    col_name <- paste0(cat, "_features")
    if (!col_name %in% names(feature_rankings)) {
      feature_rankings[[col_name]] <- ""
    }
  }

  explanation <- predictions |>
    filter(model == model_name) |>
    select(unique_id, index_nr, risk_score, risk_score_predicted) |>
    left_join(cat_contributions, by = "unique_id") |>
    left_join(feature_rankings,  by = "unique_id") |>
    mutate(model = model_name) |>
    select(model, unique_id, index_nr, risk_score, risk_score_predicted,
           all_of(categories),
           all_of(paste0(categories, "_features")))

  explanation
}


# 2. Logistic regression ----

# Contribution = scaled feature value x coefficient (additive on the log-odds).
cat("\nBuilding explanation: Logistic Regression...\n")

recipe_logistic <- fit_logistic |> extract_recipe()
model_logistic  <- fit_logistic |> extract_fit_engine()

baked_logistic <- bake(recipe_logistic, new_data = db_test)

coefs <- coef(model_logistic)
coefs <- coefs[names(coefs) != "(Intercept)"]

feature_cols <- names(coefs)

contrib_logistic <- baked_logistic |>
  select(unique_id, index_nr, all_of(feature_cols)) |>
  pivot_longer(
    cols      = all_of(feature_cols),
    names_to  = "feature",
    values_to = "value"
  ) |>
  mutate(
    coefficient  = coefs[feature],
    contribution = value * coefficient
  ) |>
  select(unique_id, index_nr, feature, contribution)

explanation_logistic <- build_explanation(
  contributions = contrib_logistic,
  predictions   = predictions_all,
  model_name    = "Logistic Regression"
)

cat("  Done. Rows:", nrow(explanation_logistic), "\n")


# 3. Elastic net ----

# Same coefficient-based contribution as logistic regression. glmnet stores a
# full lambda path; the final lambda zeroes everything out under the lasso
# penalty, so for the explanation we take the lambda with the most non-zero
# coefficients (the most informative point on the path).
cat("\nBuilding explanation: Elastic Net...\n")

library(Matrix)

recipe_elastic <- fit_elastic_net |> extract_recipe()
model_elastic  <- fit_elastic_net |> extract_fit_engine()

baked_elastic <- bake(recipe_elastic, new_data = db_test)

n_nonzero       <- Matrix::colSums(model_elastic$beta != 0)
best_lambda_idx <- which.max(n_nonzero)
best_lambda_val <- model_elastic$lambda[best_lambda_idx]

cat("  Lambda used:", best_lambda_val,
    "| Non-zero coefficients:", n_nonzero[best_lambda_idx], "\n")

coef_matrix_best <- glmnet::coef.glmnet(model_elastic, s = best_lambda_val)
coef_vec_named   <- setNames(as.numeric(coef_matrix_best), rownames(coef_matrix_best))

coef_df <- data.frame(
  feature     = names(coef_vec_named),
  coefficient = as.numeric(coef_vec_named),
  stringsAsFactors = FALSE
)
coef_df <- coef_df[coef_df$feature != "(Intercept)" & coef_df$coefficient != 0, ]

feature_cols_elastic <- coef_df$feature

contrib_elastic <- baked_elastic |>
  select(unique_id, index_nr, any_of(feature_cols_elastic)) |>
  pivot_longer(
    cols      = any_of(feature_cols_elastic),
    names_to  = "feature",
    values_to = "value"
  ) |>
  left_join(coef_df, by = "feature") |>
  mutate(contribution = value * coefficient) |>
  select(unique_id, index_nr, feature, contribution)

explanation_elastic_net <- build_explanation(
  contributions = contrib_elastic,
  predictions   = predictions_all,
  model_name    = "Elastic Net"
)

cat("  Done. Rows:", nrow(explanation_elastic_net), "\n")


# 4. Decision tree ----

# Approximation: each feature's contribution is its global rpart importance
# (normalized to sum to 1) scaled by how far the transaction's predicted risk
# sits from the global mean. Monotone with the importance ranking, scales with
# predicted risk, and tractable on the full test set.
cat("\nBuilding explanation: Decision Tree...\n")

recipe_tree <- fit_decision_tree |> extract_recipe()
model_tree  <- fit_decision_tree |> extract_fit_engine()

var_imp <- model_tree$variable.importance
if (is.null(var_imp)) {
  stop("Decision tree has no variable importance — tree may be too shallow.")
}
var_imp_norm <- var_imp / sum(var_imp)

preds_tree <- predict(fit_decision_tree,
                      new_data = db_test, type = "prob")$.pred_1

baseline <- mean(preds_tree)

contrib_tree <- map_dfr(seq_len(nrow(db_test)), function(i) {
  deviation <- preds_tree[i] - baseline
  tibble(
    unique_id    = db_test$unique_id[i],
    index_nr     = db_test$index_nr[i],
    feature      = names(var_imp_norm),
    contribution = as.numeric(var_imp_norm) * deviation
  )
})

explanation_decision_tree <- build_explanation(
  contributions = contrib_tree,
  predictions   = predictions_all,
  model_name    = "Decision Tree"
)

cat("  Done. Rows:", nrow(explanation_decision_tree), "\n")


# 5. Random forest ----

# TreeSHAP contributions via treeshap: additive, summing to the prediction.
cat("\nBuilding explanation: Random Forest (SHAP via treeshap)...\n")
cat("  This may take several minutes for 183k transactions...\n")

recipe_rf <- fit_random_forest |> extract_recipe()
model_rf  <- fit_random_forest |> extract_fit_engine()

baked_rf <- bake(recipe_rf, new_data = db_test)

predictor_cols_rf <- baked_rf |>
  select(-unique_id, -index_nr) |>
  select(where(is.numeric)) |>
  names()

X_rf <- baked_rf |>
  select(all_of(predictor_cols_rf)) |>
  as.data.frame()

unified_rf <- ranger.unify(model_rf, X_rf)
shap_rf    <- treeshap(unified_rf, X_rf, verbose = FALSE)

contrib_rf <- shap_rf$shaps |>
  as_tibble() |>
  mutate(unique_id = db_test$unique_id,
         index_nr  = db_test$index_nr) |>
  pivot_longer(
    cols      = -c(unique_id, index_nr),
    names_to  = "feature",
    values_to = "contribution"
  )

explanation_random_forest <- build_explanation(
  contributions = contrib_rf,
  predictions   = predictions_all,
  model_name    = "Random Forest"
)

cat("  Done. Rows:", nrow(explanation_random_forest), "\n")


# 6. XGBoost ----

# Same TreeSHAP approach; treeshap integrates directly with the xgb.Booster.
cat("\nBuilding explanation: XGBoost (SHAP via treeshap)...\n")
cat("  This may take several minutes for 183k transactions...\n")

recipe_xgb <- fit_xgboost |> extract_recipe()
model_xgb  <- fit_xgboost |> extract_fit_engine()

baked_xgb <- bake(recipe_xgb, new_data = db_test)

predictor_cols_xgb <- baked_xgb |>
  select(-unique_id, -index_nr) |>
  select(where(is.numeric)) |>
  names()

X_xgb <- baked_xgb |>
  select(all_of(predictor_cols_xgb)) |>
  as.data.frame()

unified_xgb <- xgboost.unify(model_xgb, X_xgb)
shap_xgb    <- treeshap(unified_xgb, X_xgb, verbose = FALSE)

contrib_xgb <- shap_xgb$shaps |>
  as_tibble() |>
  mutate(unique_id = db_test$unique_id,
         index_nr  = db_test$index_nr) |>
  pivot_longer(
    cols      = -c(unique_id, index_nr),
    names_to  = "feature",
    values_to = "contribution"
  )

explanation_xgboost <- build_explanation(
  contributions = contrib_xgb,
  predictions   = predictions_all,
  model_name    = "XGBoost"
)

cat("  Done. Rows:", nrow(explanation_xgboost), "\n")


# 7. Combine ----

cat("\nCombining explanation outputs...\n")

explanation_all <- bind_rows(
  explanation_logistic,
  explanation_elastic_net,
  explanation_decision_tree,
  explanation_random_forest,
  explanation_xgboost
)

cat("Total explanation rows:", nrow(explanation_all), "\n")


# 8. Interpretability review ----

# Top 10 and bottom 10 predicted-risk transactions per model, used to judge
# whether explanations translate into audit-relevant rationale. Two conditions
# per transaction: (1) the dominant category maps to a recognizable audit risk
# dimension, and (2) each top-3 driver is translatable into plain language. A
# model passes if both hold for at least 80% of sampled transactions.
cat("\nBuilding interpretability review samples...\n")

build_review_sample <- function(model_name, explanation_df) {
  model_expl <- explanation_df |> filter(model == model_name)

  high_risk <- model_expl |>
    arrange(desc(risk_score_predicted)) |>
    slice_head(n = 10) |>
    mutate(sample_group = "high_risk")

  low_risk <- model_expl |>
    arrange(risk_score_predicted) |>
    slice_head(n = 10) |>
    mutate(sample_group = "low_risk")

  bind_rows(high_risk, low_risk)
}

model_names <- c("Logistic Regression", "Elastic Net", "Decision Tree",
                 "Random Forest", "XGBoost")

interpretability_review <- map_dfr(model_names, function(m) {
  build_review_sample(m, explanation_all)
})

cat("Interpretability review rows:", nrow(interpretability_review),
    "(20 per model × 5 models)\n")

for (m in model_names) {
  cat("\n--- High-risk sample:", m, "---\n")
  review <- interpretability_review |>
    filter(model == m, sample_group == "high_risk") |>
    select(index_nr, risk_score, risk_score_predicted,
           `Timing Behavior`, `Amount Behavior`, `User Behavior`,
           `Journal Behavior`, `Text Flags`, `Account Flags`)
  print(review, n = 10)
}


# 9. Reproducibility check ----

# Re-predict a random 100-transaction sample and confirm the scores match those
# in predictions_all to within 1e-10 (NV COS 230 reproducibility requirement).
cat("\nRunning reproducibility check...\n")

set.seed(42)
sample_idx     <- sample(seq_len(nrow(db_test)), 100)
db_test_sample <- db_test[sample_idx, ]

check_reproducibility <- function(fit, model_name, sample_data,
                                  original_predictions, sample_idx) {
  new_preds <- predict(fit, new_data = sample_data, type = "prob")$.pred_1

  original_preds <- original_predictions |>
    filter(model == model_name) |>
    slice(sample_idx) |>
    pull(risk_score_predicted)

  tibble(
    model        = model_name,
    n_checked    = length(new_preds),
    reproducible = all(abs(new_preds - original_preds) < 1e-10),
    max_diff     = max(abs(new_preds - original_preds))
  )
}

reproducibility_check <- bind_rows(
  check_reproducibility(fit_logistic,      "Logistic Regression",
                        db_test_sample, predictions_all, sample_idx),
  check_reproducibility(fit_elastic_net,   "Elastic Net",
                        db_test_sample, predictions_all, sample_idx),
  check_reproducibility(fit_decision_tree, "Decision Tree",
                        db_test_sample, predictions_all, sample_idx),
  check_reproducibility(fit_random_forest, "Random Forest",
                        db_test_sample, predictions_all, sample_idx),
  check_reproducibility(fit_xgboost,       "XGBoost",
                        db_test_sample, predictions_all, sample_idx)
)

cat("\n--- Reproducibility Check Results ---\n")
print(reproducibility_check)
cat(sum(reproducibility_check$reproducible), "of 5 models: fully reproducible\n")


# 10. Save outputs ----

if (!dir.exists("outputs")) dir.create("outputs")

saveRDS(explanation_logistic,      "outputs/explanation_logistic.rds")
saveRDS(explanation_elastic_net,   "outputs/explanation_elastic_net.rds")
saveRDS(explanation_decision_tree, "outputs/explanation_decision_tree.rds")
saveRDS(explanation_random_forest, "outputs/explanation_random_forest.rds")
saveRDS(explanation_xgboost,       "outputs/explanation_xgboost.rds")
saveRDS(explanation_all,           "outputs/explanation_all.rds")
saveRDS(interpretability_review,   "outputs/interpretability_review.rds")
saveRDS(reproducibility_check,     "outputs/reproducibility_check.rds")

cat("\nExplanation_Interpretability.R complete — outputs written to 4. Evaluation/outputs/\n")
