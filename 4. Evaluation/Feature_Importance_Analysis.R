library(tidyverse)

# Feature importance and category-contribution analysis (SQ2: which features
# drive high-risk predictions). Three views: category contributions per model,
# feature-level importance per model, and cross-model agreement. The top 1% of
# test transactions by predicted risk is the primary lens, with full test-set
# results reported alongside.


# 0. Load data ----

cat("Loading explanation data...\n")

explanation_all <- readRDS("outputs/explanation_all.rds")
predictions_all <- readRDS("outputs/predictions_all.rds")

categories <- c("Timing Behavior", "Amount Behavior", "User Behavior",
                "Journal Behavior", "Text Flags", "Account Flags")

model_order <- c("Logistic Regression", "Elastic Net", "Decision Tree",
                 "Random Forest", "XGBoost")

cat("Explanation rows:", nrow(explanation_all), "\n")
cat("Models:", paste(unique(explanation_all$model), collapse = ", "), "\n")


# 1. Category-level contributions ----

# Mean absolute contribution per category, for the top 1% of predicted-risk
# transactions and for the full test set. Absolute values, since we want which
# categories are most active regardless of direction; share_pct makes the
# cross-model comparison easier.
cat("\nComputing category-level contributions...\n")

top1pct_unique_ids <- predictions_all |>
  group_by(model) |>
  mutate(threshold = quantile(risk_score_predicted, 0.99)) |>
  filter(risk_score_predicted >= threshold) |>
  select(model, unique_id) |>
  ungroup()

cat_contrib_top1pct <- explanation_all |>
  inner_join(top1pct_unique_ids, by = c("model", "unique_id")) |>
  group_by(model) |>
  summarise(across(all_of(categories), ~ mean(abs(.x), na.rm = TRUE)),
            .groups = "drop") |>
  pivot_longer(
    cols      = all_of(categories),
    names_to  = "category",
    values_to = "mean_abs_contribution"
  ) |>
  group_by(model) |>
  mutate(
    total        = sum(mean_abs_contribution),
    share_pct    = round(100 * mean_abs_contribution / total, 1),
    category     = factor(category, levels = categories),
    model        = factor(model,    levels = model_order)
  ) |>
  ungroup() |>
  arrange(model, desc(mean_abs_contribution))

cat_contrib_full <- explanation_all |>
  group_by(model) |>
  summarise(across(all_of(categories), ~ mean(abs(.x), na.rm = TRUE)),
            .groups = "drop") |>
  pivot_longer(
    cols      = all_of(categories),
    names_to  = "category",
    values_to = "mean_abs_contribution"
  ) |>
  group_by(model) |>
  mutate(
    total        = sum(mean_abs_contribution),
    share_pct    = round(100 * mean_abs_contribution / total, 1),
    category     = factor(category, levels = categories),
    model        = factor(model,    levels = model_order)
  ) |>
  ungroup() |>
  arrange(model, desc(mean_abs_contribution))

cat("\n--- Category Contributions: Top 1% Transactions ---\n")
cat_contrib_top1pct |>
  select(model, category, mean_abs_contribution, share_pct) |>
  print(n = 30)

cat("\n--- Category Contributions: Full Test Set ---\n")
cat_contrib_full |>
  select(model, category, mean_abs_contribution, share_pct) |>
  print(n = 30)


# 2. Feature-level importance ----

# Mean absolute contribution per feature per model, reconstructed from the
# ranked-feature strings in the explanation output. Scales differ across model
# types (log-odds vs SHAP), so rankings are read within each model, not across.
cat("\nComputing feature-level importance...\n")

feature_cols <- paste0(categories, "_features")

# Parse "1. feature (value) | 2. feature (value) | ..." into feature/contribution
parse_features <- function(feature_string) {
  if (is.na(feature_string) || feature_string == "") return(tibble(feature = character(), contribution = numeric()))
  entries <- str_split(feature_string, " \\| ")[[1]]
  map_dfr(entries, function(e) {
    m <- regmatches(e, regexec("^\\d+\\.\\s+(.+)\\s+\\((-?[0-9.e+-]+)\\)$", e))[[1]]
    if (length(m) < 3) return(tibble(feature = character(), contribution = numeric()))
    tibble(feature = trimws(m[2]), contribution = as.numeric(m[3]))
  })
}

feature_importance <- map_dfr(model_order, function(m) {
  model_expl <- explanation_all |> filter(model == m)

  map_dfr(categories, function(cat) {
    col <- paste0(cat, "_features")
    if (!col %in% names(model_expl)) return(tibble())

    map_dfr(seq_len(nrow(model_expl)), function(i) {
      parsed <- parse_features(model_expl[[col]][i])
      if (nrow(parsed) == 0) return(tibble())
      parsed |> mutate(category = cat, unique_id = model_expl$unique_id[i])
    })
  }) |>
    mutate(model = m)
}) |>
  group_by(model, category, feature) |>
  summarise(
    mean_abs_contribution = mean(abs(contribution), na.rm = TRUE),
    frequency             = n(),
    .groups               = "drop"
  ) |>
  group_by(model, category) |>
  arrange(desc(mean_abs_contribution), .by_group = TRUE) |>
  ungroup() |>
  mutate(
    model    = factor(model,    levels = model_order),
    category = factor(category, levels = categories)
  ) |>
  arrange(model, category, desc(mean_abs_contribution))

cat("\n--- All Features per Category per Model (ranked by absolute contribution) ---\n")
print(feature_importance, n = Inf)


# 3. Cross-model category ranking ----

# Average rank of each category across models (1 = most important). Consistent
# top ranks are robust findings; categories that rank high for ensembles but low
# for linear models reveal nonlinear relationships.
cat("\nComputing cross-model category rankings...\n")

category_ranks <- cat_contrib_top1pct |>
  group_by(model) |>
  mutate(rank = rank(-mean_abs_contribution, ties.method = "min")) |>
  ungroup() |>
  select(model, category, rank, share_pct)

cross_model_ranks <- category_ranks |>
  group_by(category) |>
  summarise(
    mean_rank      = round(mean(rank), 2),
    mean_share_pct = round(mean(share_pct), 1),
    ranks_by_model = paste(model_order,
                           rank[match(model_order, as.character(model))],
                           sep = "=", collapse = ", "),
    .groups        = "drop"
  ) |>
  arrange(mean_rank)

cat("\n--- Cross-Model Category Rankings (Top 1% Transactions) ---\n")
cat("(Rank 1 = most important for that model)\n\n")
print(cross_model_ranks)


# 4. Summary tables ----

# One row per category, one column per model (share_pct)
category_summary_wide <- cat_contrib_top1pct |>
  select(model, category, share_pct) |>
  pivot_wider(
    id_cols     = category,
    names_from  = model,
    values_from = share_pct
  ) |>
  left_join(
    cross_model_ranks |> select(category, mean_rank, mean_share_pct),
    by = "category"
  ) |>
  arrange(mean_rank)

cat("\n--- Summary Table: Category Share (%) by Model — Top 1% Transactions ---\n")
print(category_summary_wide)

feature_summary <- feature_importance |>
  select(model, category, feature, mean_abs_contribution, frequency) |>
  mutate(mean_abs_contribution = round(mean_abs_contribution, 4))

cat("\n--- Feature Importance Summary ---\n")
print(feature_summary, n = Inf)


# 5. Save outputs ----

saveRDS(cat_contrib_top1pct,  "outputs/category_contributions_top1pct.rds")
saveRDS(cat_contrib_full,     "outputs/category_contributions_full.rds")
saveRDS(feature_importance,   "outputs/feature_importance_per_model.rds")
saveRDS(cross_model_ranks,    "outputs/cross_model_category_ranks.rds")

write_csv(category_summary_wide, "outputs/category_contributions_summary.csv")
write_csv(feature_summary,       "outputs/feature_importance_summary.csv")

cat("\nFeature_Importance_Analysis.R complete — outputs written to 4. Evaluation/outputs/\n")
