library(tidyverse)
library(tidymodels)
library(arrow)
library(themis)
library(ranger)
library(xgboost)

# Ensemble models: random forest (ranger) and gradient boosting (xgboost), run
# under the same three imbalance strategies and 5-fold grouped CV as the
# transparent models. Class weights are passed engine-side: class.weights in
# ranger and scale_pos_weight in xgboost. The split, preprocessed training data
# and folds are reused from Modeling_simple.R so all five models are compared on
# identical data and resamples. Strategy and hyperparameters are selected on
# AUC-ROC (the custom ranking metrics don't propagate through tune_grid).


# 0. Dependencies ----

source("../2. Feature Engineering/Feature_Engineering.R")
source("Metrics.R")


# 1. Load shared objects from Modeling_simple.R ----

# Loaded rather than recomputed so both scripts use exactly the same split and
# feature definitions.
split    <- readRDS("outputs/split.rds")
db_train <- readRDS("outputs/train_preprocessed.rds")
db_test  <- testing(split)

cat("Train rows:", nrow(db_train),
    "| Flagged:", sum(db_train$risk_score == "1"),
    "(", round(100 * mean(db_train$risk_score == "1"), 2), "%)\n")

# scale_pos_weight for xgboost: ratio of negative to positive cases
n_pos            <- sum(db_train$risk_score == "1")
n_neg            <- sum(db_train$risk_score == "0")
scale_pos_weight <- n_neg / n_pos

cat("scale_pos_weight for xgboost:", round(scale_pos_weight, 1), "\n")


# 2. Base recipe ----

# Same recipe as Modeling_simple.R, rebuilt unprepped here so it can be extended
# with the SMOTE and undersampling steps (a prepped recipe can't take new steps).
recipe_base <- recipe(risk_score ~ ., data = db_train) |>

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

  step_nzv(all_predictors())             |>

  step_company_normalize(
    abs_amount,
    log_abs_amount,
    user_period_entry_count,
    user_account_period_entry_count
  ) |>

  step_normalize(
    all_numeric_predictors(),
    -abs_amount,
    -log_abs_amount,
    -user_period_entry_count,
    -user_account_period_entry_count
  )


# 3. Cross-validation folds ----

# Same seed as Modeling_simple.R so the fold assignments match across all five
# models.
set.seed(1234)

cv_folds <- group_vfold_cv(db_train, group = entry_key, v = 5)


# 4. Imbalance-strategy recipes ----

recipe_smote <- recipe_base |>
  step_smote(risk_score, over_ratio = 0.1, seed = 42)

recipe_undersample <- recipe_base |>
  step_downsample(risk_score, under_ratio = 10, seed = 42)


# 5. Random forest ----

# Tuned over mtry, trees and min_n. Class weights via ranger's class.weights;
# explanation layer uses TreeSHAP on the final fit.
rf_grid <- grid_regular(
  mtry(range  = c(2, 6)),
  trees(range = c(200, 500)),
  min_n(range = c(5, 30)),
  levels = c(mtry = 3, trees = 2, min_n = 3)
)

spec_rf_weights <- rand_forest(
  mtry  = tune(),
  trees = tune(),
  min_n = tune()
) |>
  set_engine("ranger",
             class.weights = c("0" = 1, "1" = n_neg / n_pos),
             importance    = "impurity",
             num.threads   = parallel::detectCores()) |>
  set_mode("classification")

spec_rf_standard <- rand_forest(
  mtry  = tune(),
  trees = tune(),
  min_n = tune()
) |>
  set_engine("ranger",
             importance  = "impurity",
             num.threads = parallel::detectCores()) |>
  set_mode("classification")

wf_rf_weights <- workflow() |>
  add_recipe(recipe_base) |>
  add_model(spec_rf_weights)

wf_rf_smote <- workflow() |>
  add_recipe(recipe_smote) |>
  add_model(spec_rf_standard)

wf_rf_undersample <- workflow() |>
  add_recipe(recipe_undersample) |>
  add_model(spec_rf_standard)

cat("\nTuning random forest (class weights)...\n")
cv_rf_weights <- tune_grid(
  wf_rf_weights,
  resamples = cv_folds,
  grid      = rf_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

cat("Tuning random forest (SMOTE)...\n")
cv_rf_smote <- tune_grid(
  wf_rf_smote,
  resamples = cv_folds,
  grid      = rf_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

cat("Tuning random forest (undersampling)...\n")
cv_rf_undersample <- tune_grid(
  wf_rf_undersample,
  resamples = cv_folds,
  grid      = rf_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

best_rf_weights     <- select_best(cv_rf_weights,     metric = "roc_auc")
best_rf_smote       <- select_best(cv_rf_smote,       metric = "roc_auc")
best_rf_undersample <- select_best(cv_rf_undersample, metric = "roc_auc")

rf_strategy_comparison <- bind_rows(
  show_best(cv_rf_weights,     metric = "roc_auc", n = 1) |> mutate(strategy = "weights"),
  show_best(cv_rf_smote,       metric = "roc_auc", n = 1) |> mutate(strategy = "smote"),
  show_best(cv_rf_undersample, metric = "roc_auc", n = 1) |> mutate(strategy = "undersample")
) |>
  arrange(desc(mean))

cat("\nRandom forest — strategy comparison (best AUC per strategy):\n")
print(rf_strategy_comparison |> select(strategy, mtry, trees, min_n, mean, std_err))

best_rf_strategy <- "weights"
cat("Selected strategy:", best_rf_strategy, "\n")

best_rf_params <- rf_strategy_comparison |>
  filter(strategy == best_rf_strategy) |>
  slice(1) |>
  select(mtry, trees, min_n)

best_rf_wf <- switch(best_rf_strategy,
                     "weights"     = wf_rf_weights,
                     "smote"       = wf_rf_smote,
                     "undersample" = wf_rf_undersample
)

best_rf_wf <- finalize_workflow(best_rf_wf, best_rf_params)

cat("Fitting final random forest on full training set...\n")
fit_random_forest <- fit(best_rf_wf, data = db_train)

results_random_forest <- list(
  weights       = cv_rf_weights,
  smote         = cv_rf_smote,
  undersample   = cv_rf_undersample,
  comparison    = rf_strategy_comparison,
  best_strategy = best_rf_strategy,
  best_params   = best_rf_params
)


# 6. Gradient boosting (XGBoost) ----

# Trees are added sequentially to correct earlier residuals. Class weights via
# scale_pos_weight; explanation layer uses TreeSHAP.
xgb_grid <- grid_regular(
  trees(range          = c(100, 500)),
  tree_depth(range     = c(3, 8)),
  learn_rate(range     = c(-3, -1), trans = log10_trans()),
  loss_reduction(range = c(-5, 0),  trans = log10_trans()),
  sample_prop(range    = c(0.5, 1.0)),
  levels = 2
)

spec_xgb_weights <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune(),
  sample_size    = tune()
) |>
  set_engine("xgboost",
             scale_pos_weight = scale_pos_weight,
             nthread          = parallel::detectCores()) |>
  set_mode("classification")

spec_xgb_standard <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune(),
  sample_size    = tune()
) |>
  set_engine("xgboost",
             nthread = parallel::detectCores()) |>
  set_mode("classification")

wf_xgb_weights <- workflow() |>
  add_recipe(recipe_base) |>
  add_model(spec_xgb_weights)

wf_xgb_smote <- workflow() |>
  add_recipe(recipe_smote) |>
  add_model(spec_xgb_standard)

wf_xgb_undersample <- workflow() |>
  add_recipe(recipe_undersample) |>
  add_model(spec_xgb_standard)

cat("\nTuning xgboost (class weights)...\n")
cv_xgb_weights <- tune_grid(
  wf_xgb_weights,
  resamples = cv_folds,
  grid      = xgb_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

cat("Tuning xgboost (SMOTE)...\n")
cv_xgb_smote <- tune_grid(
  wf_xgb_smote,
  resamples = cv_folds,
  grid      = xgb_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

cat("Tuning xgboost (undersampling)...\n")
cv_xgb_undersample <- tune_grid(
  wf_xgb_undersample,
  resamples = cv_folds,
  grid      = xgb_grid,
  metrics   = metric_set(roc_auc),
  control   = control_grid(save_pred = TRUE)
)

best_xgb_weights     <- select_best(cv_xgb_weights,     metric = "roc_auc")
best_xgb_smote       <- select_best(cv_xgb_smote,       metric = "roc_auc")
best_xgb_undersample <- select_best(cv_xgb_undersample, metric = "roc_auc")

xgb_strategy_comparison <- bind_rows(
  show_best(cv_xgb_weights,     metric = "roc_auc", n = 1) |> mutate(strategy = "weights"),
  show_best(cv_xgb_smote,       metric = "roc_auc", n = 1) |> mutate(strategy = "smote"),
  show_best(cv_xgb_undersample, metric = "roc_auc", n = 1) |> mutate(strategy = "undersample")
) |>
  arrange(desc(mean))

cat("\nXGBoost — strategy comparison (best AUC per strategy):\n")
print(xgb_strategy_comparison |>
        select(strategy, trees, tree_depth, learn_rate, mean, std_err))

best_xgb_strategy <- "weights"
cat("Selected strategy:", best_xgb_strategy, "\n")

best_xgb_params <- xgb_strategy_comparison |>
  filter(strategy == best_xgb_strategy) |>
  slice(1) |>
  select(trees, tree_depth, learn_rate, loss_reduction, sample_size)

best_xgb_wf <- switch(best_xgb_strategy,
                      "weights"     = wf_xgb_weights,
                      "smote"       = wf_xgb_smote,
                      "undersample" = wf_xgb_undersample
)

best_xgb_wf <- finalize_workflow(best_xgb_wf, best_xgb_params)

cat("Fitting final xgboost on full training set...\n")
fit_xgboost <- fit(best_xgb_wf, data = db_train)

results_xgboost <- list(
  weights       = cv_xgb_weights,
  smote         = cv_xgb_smote,
  undersample   = cv_xgb_undersample,
  comparison    = xgb_strategy_comparison,
  best_strategy = best_xgb_strategy,
  best_params   = best_xgb_params
)


# 7. Save outputs ----

if (!dir.exists("outputs")) dir.create("outputs")

saveRDS(results_random_forest, "outputs/results_random_forest.rds")
saveRDS(results_xgboost,       "outputs/results_xgboost.rds")
saveRDS(fit_random_forest,     "outputs/fit_random_forest.rds")
saveRDS(fit_xgboost,           "outputs/fit_xgboost.rds")

cat("\nModeling_complex.R complete — outputs written to 3. Data Modeling/outputs/\n")
