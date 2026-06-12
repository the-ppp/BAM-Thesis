library(tidymodels)
library(yardstick)

# Custom ranking metrics for the audit use case. Auditors only review the
# top-ranked transactions, so what matters is how well the model concentrates
# flagged entries at the top rather than overall discrimination. k is a
# proportion of the data (not a fixed count) so it means the same thing across
# folds and across the train/test sets. AUC-ROC is left to yardstick and used
# as a secondary measure.


# ---- Core computations ----
# Operate on a numeric truth vector (0/1) and predicted probabilities. Non-finite
# predictions (e.g. from a non-converged glm) are pushed to the bottom of the
# ranking rather than throwing.

compute_precision_at_k <- function(truth, estimate, k_prop = 0.05) {
  estimate <- ifelse(is.finite(estimate), estimate, 0)
  n     <- length(truth)
  k     <- max(1L, floor(n * k_prop))
  top_k <- truth[order(estimate, decreasing = TRUE)][1:k]
  mean(top_k)
}

# Precision@k over the overall flag rate. Returns 0 if the flag rate is 0/NA.
compute_lift_at_k <- function(truth, estimate, k_prop = 0.05) {
  baseline <- mean(truth, na.rm = TRUE)
  if (is.na(baseline) || baseline == 0) return(0)
  compute_precision_at_k(truth, estimate, k_prop) / baseline
}

# Normalized DCG@k: rewards placing flagged entries higher within the top k.
compute_ndcg_at_k <- function(truth, estimate, k_prop = 0.05) {
  estimate <- ifelse(is.finite(estimate), estimate, 0)
  n     <- length(truth)
  k     <- max(1L, floor(n * k_prop))
  top_k <- truth[order(estimate, decreasing = TRUE)][1:k]

  dcg   <- sum(top_k / log2(seq_along(top_k) + 1))

  n_pos <- sum(truth)
  if (n_pos == 0) return(0)
  ideal <- c(rep(1, min(n_pos, k)), rep(0, max(0, k - n_pos)))
  idcg  <- sum(ideal / log2(seq_along(ideal) + 1))

  if (idcg == 0) return(0)
  dcg / idcg
}


# ---- yardstick wrappers (default k = 0.05) ----
# truth must be a factor with the positive class as the second level, so
# as.numeric(truth) - 1 maps it to 0/1. Estimate is the .pred_1 column.

precision_at_k <- function(data, ...) {
  UseMethod("precision_at_k")
}
precision_at_k <- new_prob_metric(precision_at_k, direction = "maximize")

precision_at_k.data.frame <- function(data, truth, estimate,
                                      k_prop = 0.05, na_rm = TRUE, ...) {
  truth_col    <- data[[rlang::as_name(rlang::ensym(truth))]]
  estimate_col <- data[[".pred_1"]]
  value        <- precision_at_k_vec(truth_col, estimate_col,
                                     k_prop = k_prop, na_rm = na_rm)
  tibble(.metric = "precision_at_k", .estimator = "binary", .estimate = value)
}

precision_at_k_vec <- function(truth, estimate, k_prop = 0.05,
                               na_rm = TRUE, ...) {
  if (na_rm) {
    complete <- !is.na(truth) & !is.na(estimate)
    truth    <- truth[complete]
    estimate <- estimate[complete]
  }
  compute_precision_at_k(as.numeric(truth) - 1, estimate, k_prop)
}

lift_at_k <- function(data, ...) {
  UseMethod("lift_at_k")
}
lift_at_k <- new_prob_metric(lift_at_k, direction = "maximize")

lift_at_k.data.frame <- function(data, truth, estimate,
                                 k_prop = 0.05, na_rm = TRUE, ...) {
  truth_col    <- data[[rlang::as_name(rlang::ensym(truth))]]
  estimate_col <- data[[".pred_1"]]
  value        <- lift_at_k_vec(truth_col, estimate_col,
                                k_prop = k_prop, na_rm = na_rm)
  tibble(.metric = "lift_at_k", .estimator = "binary", .estimate = value)
}

lift_at_k_vec <- function(truth, estimate, k_prop = 0.05,
                          na_rm = TRUE, ...) {
  if (na_rm) {
    complete <- !is.na(truth) & !is.na(estimate)
    truth    <- truth[complete]
    estimate <- estimate[complete]
  }
  compute_lift_at_k(as.numeric(truth) - 1, estimate, k_prop)
}

ndcg_at_k <- function(data, ...) {
  UseMethod("ndcg_at_k")
}
ndcg_at_k <- new_prob_metric(ndcg_at_k, direction = "maximize")

ndcg_at_k.data.frame <- function(data, truth, estimate,
                                 k_prop = 0.05, na_rm = TRUE, ...) {
  truth_col    <- data[[rlang::as_name(rlang::ensym(truth))]]
  estimate_col <- data[[".pred_1"]]
  value        <- ndcg_at_k_vec(truth_col, estimate_col,
                                k_prop = k_prop, na_rm = na_rm)
  tibble(.metric = "ndcg_at_k", .estimator = "binary", .estimate = value)
}

ndcg_at_k_vec <- function(truth, estimate, k_prop = 0.05,
                          na_rm = TRUE, ...) {
  if (na_rm) {
    complete <- !is.na(truth) & !is.na(estimate)
    truth    <- truth[complete]
    estimate <- estimate[complete]
  }
  compute_ndcg_at_k(as.numeric(truth) - 1, estimate, k_prop)
}


# ---- Wrappers at k = 0.01 and k = 0.10 for the robustness checks ----

precision_at_k_01 <- function(data, ...) UseMethod("precision_at_k_01")
precision_at_k_01 <- new_prob_metric(precision_at_k_01, direction = "maximize")
precision_at_k_01.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  truth_col    <- data[[as_label(enquo(truth))]]
  estimate_col <- data[[as_label(enquo(estimate))]]
  value        <- precision_at_k_vec(truth_col, estimate_col, k_prop = 0.01, na_rm = na_rm)
  tibble(.metric = "precision_at_k_01", .estimator = "binary", .estimate = value)
}

precision_at_k_10 <- function(data, ...) UseMethod("precision_at_k_10")
precision_at_k_10 <- new_prob_metric(precision_at_k_10, direction = "maximize")
precision_at_k_10.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  truth_col    <- data[[as_label(enquo(truth))]]
  estimate_col <- data[[as_label(enquo(estimate))]]
  value        <- precision_at_k_vec(truth_col, estimate_col, k_prop = 0.10, na_rm = na_rm)
  tibble(.metric = "precision_at_k_10", .estimator = "binary", .estimate = value)
}

lift_at_k_01 <- function(data, ...) UseMethod("lift_at_k_01")
lift_at_k_01 <- new_prob_metric(lift_at_k_01, direction = "maximize")
lift_at_k_01.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  truth_col    <- data[[as_label(enquo(truth))]]
  estimate_col <- data[[as_label(enquo(estimate))]]
  value        <- lift_at_k_vec(truth_col, estimate_col, k_prop = 0.01, na_rm = na_rm)
  tibble(.metric = "lift_at_k_01", .estimator = "binary", .estimate = value)
}

lift_at_k_10 <- function(data, ...) UseMethod("lift_at_k_10")
lift_at_k_10 <- new_prob_metric(lift_at_k_10, direction = "maximize")
lift_at_k_10.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  truth_col    <- data[[as_label(enquo(truth))]]
  estimate_col <- data[[as_label(enquo(estimate))]]
  value        <- lift_at_k_vec(truth_col, estimate_col, k_prop = 0.10, na_rm = na_rm)
  tibble(.metric = "lift_at_k_10", .estimator = "binary", .estimate = value)
}

ndcg_at_k_01 <- function(data, ...) UseMethod("ndcg_at_k_01")
ndcg_at_k_01 <- new_prob_metric(ndcg_at_k_01, direction = "maximize")
ndcg_at_k_01.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  truth_col    <- data[[as_label(enquo(truth))]]
  estimate_col <- data[[as_label(enquo(estimate))]]
  value        <- ndcg_at_k_vec(truth_col, estimate_col, k_prop = 0.01, na_rm = na_rm)
  tibble(.metric = "ndcg_at_k_01", .estimator = "binary", .estimate = value)
}

ndcg_at_k_10 <- function(data, ...) UseMethod("ndcg_at_k_10")
ndcg_at_k_10 <- new_prob_metric(ndcg_at_k_10, direction = "maximize")
ndcg_at_k_10.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  truth_col    <- data[[as_label(enquo(truth))]]
  estimate_col <- data[[as_label(enquo(estimate))]]
  value        <- ndcg_at_k_vec(truth_col, estimate_col, k_prop = 0.10, na_rm = na_rm)
  tibble(.metric = "ndcg_at_k_10", .estimator = "binary", .estimate = value)
}


# Default metric set used during cross-validation tuning
audit_metrics <- metric_set(
  roc_auc,
  precision_at_k,
  precision_at_k_01,
  lift_at_k,
  ndcg_at_k
)
