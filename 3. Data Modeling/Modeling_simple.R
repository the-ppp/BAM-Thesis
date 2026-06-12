library(tidyverse)
library(tidymodels)
library(arrow)
library(themis)
library(rpart)
library(glmnet)

# Transparent models: logistic regression, elastic net, and a depth-limited
# decision tree. Each is run under three imbalance strategies (class weights,
# SMOTE, undersampling) with 5-fold grouped CV, and the best strategy per model
# is carried to a final fit on the full training set. Strategy and
# hyperparameter selection use AUC-ROC; the ranking metrics are reported later
# in Evaluation.R. The split, preprocessed training data, and prepped recipe are
# written to outputs/ for the ensemble script and the explanation layer to reuse.


# 0. Dependencies ----

source("../2. Feature Engineering/Feature_Engineering.R")
source("Metrics.R")


# 1. Load and preprocess ----

db <- read_parquet("../0. Data/GL_Dataset.parquet")

# Row-level features and flags carry no group statistics, so they are computed
# on the full dataset before the split.
db_preprocessed <- db |>
  add_static_features() |>
  add_rule_flags() |>
  mutate(entry_key = paste0(source_dataset, "_", journal_entry_number))

# Positive class ("1") as the second factor level (tidymodels convention, also
# assumed by the custom metrics in Metrics.R).
db_preprocessed <- db_preprocessed |>
  mutate(risk_score = factor(risk_score, levels = c("0", "1")))


# 2. Train/test split ----

# Grouped by entry_key so every GL line of a journal entry stays on one side;
# splitting siblings would leak double-entry information. group_initial_split
# can't stratify, so balance is checked below. The test set is locked here and
# not touched again until Evaluation.R.
set.seed(123)

split    <- group_initial_split(db_preprocessed, group = entry_key, prop = 0.8)
db_train <- training(split)
db_test  <- testing(split)

cat("Train rows:", nrow(db_train),
    "| Flagged:", sum(db_train$risk_score == "1"),
    "(", round(100 * mean(db_train$risk_score == "1"), 2), "%)\n")
cat("Test rows: ", nrow(db_test),
    "| Flagged:", sum(db_test$risk_score == "1"),
    "(", round(100 * mean(db_test$risk_score == "1"), 2), "%)\n")

db_train <- db_train |>
  mutate(
    case_weights = importance_weights(
      if_else(risk_score == "1",
              1 / mean(risk_score == "1"),
              1)
    )
  )


# 3. Base recipe ----

# Engineered features carry the behavioural signal in a company-relative form,
# so the raw categorical columns (company, journal, user, gl_account, ...) move
# to an ID role: kept in the data for the explanation layer but out of the model
# matrix, which stops the models memorising company-specific structure. gl_type
# is the only raw column kept as a predictor. Normalization matters for the
# linear models; the tree ignores scale but shares the recipe for comparability.
recipe_base <- recipe(risk_score ~ ., data = db_train) |>

  # Leakage-safe group-statistic features (learned from the training fold)
  step_timing_features(role = "predictor")          |>
  step_amount_features(role = "predictor")          |>
  step_user_features(role = "predictor")            |>
  step_journal_features(role = "predictor")         |>

  update_role(index_nr, unique_id, entry_key, risk_reason, entry_description,
              post_date, entry_date, source_dataset, document,
              gl_description, relation_name, journal_entry_number,
              company, journal, user, relation_type, gl_account,
              cost_center, amount, gl_period,
              new_role = "ID") |>

  step_novel(all_nominal_predictors())   |>
  step_unknown(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors())   |>

  step_nzv(all_predictors()) |>

  # Per-company normalization for the size-dependent features
  step_company_normalize(
    abs_amount,
    log_abs_amount,
    user_period_entry_count,
    user_account_period_entry_count
  ) |>

  # Global normalization for the rest (the four above are already standardized)
  step_normalize(
    all_numeric_predictors(),
    -abs_amount,
    -log_abs_amount,
    -user_period_entry_count,
    -user_account_period_entry_count
  )


# 4. Cross-validation folds ----

# Same folds reused across every model and strategy so comparisons are
# like-for-like. Grouped by entry_key; balance verified in the report below.
set.seed(1234)

cv_folds <- group_vfold_cv(db_train, group = entry_key, v = 5)


# 4b. Balance check ----

# Grouped resampling does not stratify, so confirm manually that no company or
# fold is left short of positives and that no entry_key leaks across train/test.
cat("\n====== BALANCE VERIFICATION REPORT ======\n")

cat("\n--- Train / Test split ---\n")

split_per_company <- bind_rows(
  db_train |>
    group_by(company) |>
    summarise(n = n(), n_flag = sum(risk_score == "1"),
              flag_pct = round(100 * mean(risk_score == "1"), 2),
              .groups = "drop") |>
    mutate(set = "train"),
  db_test |>
    group_by(company) |>
    summarise(n = n(), n_flag = sum(risk_score == "1"),
              flag_pct = round(100 * mean(risk_score == "1"), 2),
              .groups = "drop") |>
    mutate(set = "test")
) |>
  select(set, company, n, n_flag, flag_pct) |>
  arrange(set, company)

cat("\nPer-company balance:\n")
print(split_per_company, n = Inf)

split_overall <- bind_rows(
  tibble(set = "train", n = nrow(db_train),
         n_flag   = sum(db_train$risk_score == "1"),
         flag_pct = round(100 * mean(db_train$risk_score == "1"), 2)),
  tibble(set = "test",  n = nrow(db_test),
         n_flag   = sum(db_test$risk_score == "1"),
         flag_pct = round(100 * mean(db_test$risk_score == "1"), 2))
)

cat("\nOverall balance:\n")
print(split_overall)

n_leaked <- length(intersect(db_train$entry_key, db_test$entry_key))
cat("\nEntry key overlap (train vs test):", n_leaked,
    if (n_leaked == 0) "-- no leakage" else "-- LEAKAGE DETECTED", "\n")

cat("\n--- CV fold balance (assessment sets) ---\n")

fold_per_company <- purrr::imap_dfr(cv_folds$splits, function(sp, i) {
  assessment(sp) |>
    group_by(company) |>
    summarise(n = n(), n_flag = sum(risk_score == "1"),
              flag_pct = round(100 * mean(risk_score == "1"), 2),
              .groups = "drop") |>
    mutate(fold = i)
}) |>
  select(fold, company, n, n_flag, flag_pct) |>
  arrange(fold, company)

cat("\nPer-company per-fold (assessment set):\n")
print(fold_per_company, n = Inf)

fold_overall <- purrr::imap_dfr(cv_folds$splits, function(sp, i) {
  assess <- assessment(sp)
  tibble(fold = i, n = nrow(assess),
         n_flag   = sum(assess$risk_score == "1"),
         flag_pct = round(100 * mean(assess$risk_score == "1"), 2))
})

cat("\nPer-fold overall (assessment set):\n")
print(fold_overall, n = Inf)

cat("\n====== END BALANCE VERIFICATION ======\n\n")


# 5. Imbalance-strategy recipes ----

# SMOTE and downsampling sit after the feature steps, so resampling happens in
# the engineered feature space rather than the raw input space. Class weights
# need no recipe change and are applied at the workflow level.
recipe_smote <- recipe_base |>
  step_smote(risk_score, over_ratio = 0.1, seed = 42)

recipe_undersample <- recipe_base |>
  step_downsample(risk_score, under_ratio = 10, seed = 42)


# 6. Logistic regression ----

# Transparent baseline. Explanation-layer contributions come straight from the
# coefficients (scaled feature value x coefficient).
spec_logistic_weights <- logistic_reg() |>
  set_engine("glm", family = binomial(link = "logit")) |>
  set_mode("classification")

spec_logistic_standard <- logistic_reg() |>
  set_engine("glm", family = binomial(link = "logit")) |>
  set_mode("classification")

wf_logistic_weights <- workflow() |>
  add_recipe(recipe_base) |>
  add_model(spec_logistic_weights) |>
  add_case_weights(case_weights)

wf_logistic_smote <- workflow() |>
  add_recipe(recipe_smote) |>
  add_model(spec_logistic_standard)

wf_logistic_undersample <- workflow() |>
  add_recipe(recipe_undersample) |>
  add_model(spec_logistic_standard)

cat("\nFitting logistic regression (class weights)...\n")
cv_logistic_weights <- fit_resamples(
  wf_logistic_weights,
  resamples = cv_folds,
  metrics   = audit_metrics,
  control   = control_resamples(save_pred = TRUE)
)

cat("Fitting logistic regression (SMOTE)...\n")
cv_logistic_smote <- fit_resamples(
  wf_logistic_smote,
  resamples = cv_folds,
  metrics   = audit_metrics,
  control   = control_resamples(save_pred = TRUE)
)

cat("Fitting logistic regression (undersampling)...\n")
cv_logistic_undersample <- fit_resamples(
  wf_logistic_undersample,
  resamples = cv_folds,
  metrics   = audit_metrics,
  control   = control_resamples(save_pred = TRUE)
)

logistic_strategy_comparison <- bind_rows(
  collect_metrics(cv_logistic_weights)     |> mutate(strategy = "weights"),
  collect_metrics(cv_logistic_smote)       |> mutate(strategy = "smote"),
  collect_metrics(cv_logistic_undersample) |> mutate(strategy = "undersample")
) |>
  filter(.metric == "roc_auc") |>
  arrange(desc(mean))

cat("\nLogistic regression — strategy comparison (AUC-ROC):\n")
print(logistic_strategy_comparison |> select(strategy, mean, std_err))

# Class weighting ranked first or within one standard error of the top strategy
# in every run, and is selected uniformly across all five models.
best_logistic_strategy <- "weights"
cat("Selected strategy:", best_logistic_strategy, "\n")

best_logistic_wf <- switch(best_logistic_strategy,
                           "weights"     = wf_logistic_weights,
                           "smote"       = wf_logistic_smote,
                           "undersample" = wf_logistic_undersample
)

cat("Fitting final logistic regression on full training set...\n")
fit_logistic <- fit(best_logistic_wf, data = db_train)

results_logistic <- list(
  weights       = cv_logistic_weights,
  smote         = cv_logistic_smote,
  undersample   = cv_logistic_undersample,
  comparison    = logistic_strategy_comparison,
  best_strategy = best_logistic_strategy
)


# 7. Elastic net ----

# Regularized logistic regression; penalty and mixture tuned on AUC-ROC.
# tune_grid uses roc_auc rather than audit_metrics because the custom prob
# metrics do not propagate the tuned hyperparameter values through tune_grid's
# result aggregation. Contributions match logistic regression, with zeroed
# coefficients contributing nothing.
elastic_net_grid <- grid_regular(
  penalty(range = c(-4, 0), trans = log10_trans()),
  mixture(range = c(0, 1)),
  levels = c(penalty = 20, mixture = 5)
)

spec_elastic_net <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) |>
  set_engine("glmnet") |>
  set_mode("classification")

wf_elastic_weights <- workflow() |>
  add_recipe(recipe_base) |>
  add_model(spec_elastic_net) |>
  add_case_weights(case_weights)

wf_elastic_smote <- workflow() |>
  add_recipe(recipe_smote) |>
  add_model(spec_elastic_net)

wf_elastic_undersample <- workflow() |>
  add_recipe(recipe_undersample) |>
  add_model(spec_elastic_net)

cat("\nTuning elastic net (class weights)...\n")
cv_elastic_weights <- tune_grid(
  wf_elastic_weights,
  resamples = cv_folds,
  grid      = elastic_net_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

cat("Tuning elastic net (SMOTE)...\n")
cv_elastic_smote <- tune_grid(
  wf_elastic_smote,
  resamples = cv_folds,
  grid      = elastic_net_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

cat("Tuning elastic net (undersampling)...\n")
cv_elastic_undersample <- tune_grid(
  wf_elastic_undersample,
  resamples = cv_folds,
  grid      = elastic_net_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

best_elastic_weights     <- select_best(cv_elastic_weights,     metric = "roc_auc")
best_elastic_smote       <- select_best(cv_elastic_smote,       metric = "roc_auc")
best_elastic_undersample <- select_best(cv_elastic_undersample, metric = "roc_auc")

elastic_strategy_comparison <- bind_rows(
  show_best(cv_elastic_weights,     metric = "roc_auc", n = 1) |> mutate(strategy = "weights"),
  show_best(cv_elastic_smote,       metric = "roc_auc", n = 1) |> mutate(strategy = "smote"),
  show_best(cv_elastic_undersample, metric = "roc_auc", n = 1) |> mutate(strategy = "undersample")
) |>
  arrange(desc(mean))

cat("\nElastic net — strategy comparison (best AUC per strategy):\n")
print(elastic_strategy_comparison |> select(strategy, penalty, mixture, mean, std_err))

best_elastic_strategy <- "weights"
cat("Selected strategy:", best_elastic_strategy, "\n")

best_elastic_params <- elastic_strategy_comparison |>
  filter(strategy == best_elastic_strategy) |>
  slice(1) |>
  select(penalty, mixture)

best_elastic_wf <- switch(best_elastic_strategy,
                          "weights"     = wf_elastic_weights,
                          "smote"       = wf_elastic_smote,
                          "undersample" = wf_elastic_undersample
)

best_elastic_wf <- finalize_workflow(best_elastic_wf, best_elastic_params)

cat("Fitting final elastic net on full training set...\n")
fit_elastic_net <- fit(best_elastic_wf, data = db_train)

results_elastic_net <- list(
  weights       = cv_elastic_weights,
  smote         = cv_elastic_smote,
  undersample   = cv_elastic_undersample,
  comparison    = elastic_strategy_comparison,
  best_strategy = best_elastic_strategy,
  best_params   = best_elastic_params
)


# 8. Decision tree ----

# Depth is capped (4-10) so the tree stays documentable in an audit context.
# Tuned on AUC-ROC for the same reason as elastic net.
tree_grid <- grid_regular(
  tree_depth(range = c(4, 10)),
  cost_complexity(range = c(-4, -1), trans = log10_trans()),
  levels = c(tree_depth = 4, cost_complexity = 5)
)

spec_tree_weights <- decision_tree(
  tree_depth      = tune(),
  cost_complexity = tune()
) |>
  set_engine("rpart") |>
  set_mode("classification")

spec_tree_standard <- decision_tree(
  tree_depth      = tune(),
  cost_complexity = tune()
) |>
  set_engine("rpart") |>
  set_mode("classification")

wf_tree_weights <- workflow() |>
  add_recipe(recipe_base) |>
  add_model(spec_tree_weights) |>
  add_case_weights(case_weights)

wf_tree_smote <- workflow() |>
  add_recipe(recipe_smote) |>
  add_model(spec_tree_standard)

wf_tree_undersample <- workflow() |>
  add_recipe(recipe_undersample) |>
  add_model(spec_tree_standard)

cat("\nTuning decision tree (class weights)...\n")
cv_tree_weights <- tune_grid(
  wf_tree_weights,
  resamples = cv_folds,
  grid      = tree_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

cat("Tuning decision tree (SMOTE)...\n")
cv_tree_smote <- tune_grid(
  wf_tree_smote,
  resamples = cv_folds,
  grid      = tree_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

cat("Tuning decision tree (undersampling)...\n")
cv_tree_undersample <- tune_grid(
  wf_tree_undersample,
  resamples = cv_folds,
  grid      = tree_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

best_tree_weights     <- select_best(cv_tree_weights,     metric = "roc_auc")
best_tree_smote       <- select_best(cv_tree_smote,       metric = "roc_auc")
best_tree_undersample <- select_best(cv_tree_undersample, metric = "roc_auc")

tree_strategy_comparison <- bind_rows(
  show_best(cv_tree_weights,     metric = "roc_auc", n = 1) |> mutate(strategy = "weights"),
  show_best(cv_tree_smote,       metric = "roc_auc", n = 1) |> mutate(strategy = "smote"),
  show_best(cv_tree_undersample, metric = "roc_auc", n = 1) |> mutate(strategy = "undersample")
) |>
  arrange(desc(mean))

cat("\nDecision tree — strategy comparison (best AUC per strategy):\n")
print(tree_strategy_comparison |> select(strategy, tree_depth, cost_complexity, mean, std_err))

best_tree_strategy <- "weights"
cat("Selected strategy:", best_tree_strategy, "\n")
cat("Selected tree depth:", tree_strategy_comparison$tree_depth[1], "\n")

best_tree_params <- tree_strategy_comparison |>
  filter(strategy == best_tree_strategy) |>
  slice(1) |>
  select(tree_depth, cost_complexity)

best_tree_wf <- switch(best_tree_strategy,
                       "weights"     = wf_tree_weights,
                       "smote"       = wf_tree_smote,
                       "undersample" = wf_tree_undersample
)

best_tree_wf <- finalize_workflow(best_tree_wf, best_tree_params)

cat("Fitting final decision tree on full training set...\n")
fit_decision_tree <- fit(best_tree_wf, data = db_train)

results_decision_tree <- list(
  weights       = cv_tree_weights,
  smote         = cv_tree_smote,
  undersample   = cv_tree_undersample,
  comparison    = tree_strategy_comparison,
  best_strategy = best_tree_strategy,
  best_params   = best_tree_params
)


# 9. Save outputs ----

if (!dir.exists("outputs")) dir.create("outputs")

# Split, preprocessed data and prepped recipe — reused by Modeling_complex.R
# and the explanation layer
saveRDS(split,    "outputs/split.rds")
saveRDS(db_train, "outputs/train_preprocessed.rds")

recipe_prepped <- prep(recipe_base, training = db_train)
saveRDS(recipe_prepped, "outputs/recipe_base_prepped.rds")

saveRDS(results_logistic,      "outputs/results_logistic.rds")
saveRDS(results_elastic_net,   "outputs/results_elastic_net.rds")
saveRDS(results_decision_tree, "outputs/results_decision_tree.rds")

saveRDS(fit_logistic,      "outputs/fit_logistic.rds")
saveRDS(fit_elastic_net,   "outputs/fit_elastic_net.rds")
saveRDS(fit_decision_tree, "outputs/fit_decision_tree.rds")

cat("\nModeling_simple.R complete — outputs written to 3. Data Modeling/outputs/\n")
