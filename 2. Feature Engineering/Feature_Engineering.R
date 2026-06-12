library(tidyverse)
library(tidymodels)
library(stringr)
library(stringdist)
library(purrr)
library(lubridate)

# Feature engineering definitions for the risk scoring model. Sourcing this
# file only registers functions; nothing runs until they are called.
#
# Row-level features (add_static_features, add_rule_flags) carry no group
# statistics and are applied to the full dataset before the split. The four
# custom recipe steps learn their lookups from the training fold during prep()
# and join them on during bake(), so no statistic is ever computed from
# validation or test data. Unseen groups at bake time fall back to safe values:
# 1 for count features, 1.0 for rarity scores (maximally rare), and the global
# training mean for rate features.


# ---- Row-level features ----

add_static_features <- function(db) {

  db |>

    mutate(
      is_weekend_post   = if_else(wday(post_date)  %in% c(1L, 7L), 1L, 0L),
      is_weekend_entry  = if_else(wday(entry_date) %in% c(1L, 7L), 1L, 0L)
    ) |>

    # Period-12 entries made on the last calendar day of the month. Uses
    # entry_date rather than post_date, since some ERPs set post_date to the
    # period-opening date (e.g. the 1st), which carries no within-period signal.
    mutate(
      is_last_fiscal_day = if_else(
        gl_period == 12 &
          as.Date(entry_date) == (ceiling_date(as.Date(entry_date), unit = "month") - days(1)),
        1L, 0L
      )
    ) |>

    # Positive = entry created after posting (retroactive); negative = pre-dated
    mutate(
      days_between_entry_and_posting = as.numeric(
        difftime(entry_date, post_date, units = "days")
      )
    ) |>

    mutate(
      abs_amount     = abs(amount),
      log_abs_amount = log1p(abs(amount))
    ) |>

    mutate(
      is_round_1000 = if_else(abs(abs_amount %% 1000) < 1e-6, 1L, 0L)
    )
}


# ---- Account flag lists ----
# Each unique gl_account is tagged as revenue, accrual, or provision (or left
# blank) in a classification CSV kept alongside the dataset. Codes are
# company-specific, so matching is on the full gl_account value.

.account_classification_path <- "~/Desktop/BAM - RSM/Thesis/Work_Material/0. Data/accounts_classified.csv"

if (!file.exists(.account_classification_path)) {
  stop(
    "Account classification file not found at:\n  ",
    .account_classification_path
  )
}

.account_classification <- readr::read_csv(
  .account_classification_path,
  show_col_types = FALSE
)

revenue_accounts <- .account_classification |>
  dplyr::filter(classification == "revenue") |>
  dplyr::pull(gl_account)

accrual_accounts <- .account_classification |>
  dplyr::filter(classification == "accrual") |>
  dplyr::pull(gl_account)

provision_accounts <- .account_classification |>
  dplyr::filter(classification == "provision") |>
  dplyr::pull(gl_account)

cat(
  "Loaded account flag lists from:\n  ", .account_classification_path, "\n",
  "  revenue_accounts   : ", length(revenue_accounts),   " accounts\n",
  "  accrual_accounts   : ", length(accrual_accounts),   " accounts\n",
  "  provision_accounts : ", length(provision_accounts), " accounts\n",
  sep = ""
)


# ---- Text and account flags ----

.safe_lower <- function(x) {
  str_to_lower(if_else(is.na(x), "", x))
}

# Single-word match with Levenshtein tolerance for typos
.contains_approx_word <- function(text, keywords, max_dist = 1) {
  text <- .safe_lower(text)
  sapply(text, function(x) {
    words <- unlist(str_split(x, "\\s+"))
    words <- words[words != ""]
    any(map_lgl(keywords, ~ any(stringdist(words, .x, method = "lv") <= max_dist)))
  }, USE.NAMES = FALSE)
}

# Exact regex phrase match
.contains_phrase <- function(text, pattern) {
  str_detect(.safe_lower(text), pattern)
}

# All-zero vector if the account list is empty
.make_account_flag <- function(account_vector, valid_accounts) {
  if (length(valid_accounts) == 0) {
    rep(0L, length(account_vector))
  } else {
    if_else(account_vector %in% valid_accounts, 1L, 0L)
  }
}

.kw_correction   <- c("correctie", "correction", "adjustment")
.kw_accrual      <- c("accrual")
.kw_bonus        <- c("bonus")
.kw_valuation    <- c("valuation")
.kw_equity       <- c("share", "dividend")
.kw_batch        <- c("batch")

.pat_accrual_unbilled   <- "niet\\s+gefactureerd"
.pat_internal_transfer  <- "internal\\s+transfer"
.pat_credit_card        <- "credit\\s+card"
.pat_payroll_tax        <- "payroll\\s+tax"

add_rule_flags <- function(db) {

  db |>

    mutate(
      flag_correction_adjustment = if_else(
        .contains_approx_word(entry_description, .kw_correction, max_dist = 1),
        1L, 0L
      ),

      flag_accrual_unbilled = if_else(
        .contains_approx_word(entry_description, .kw_accrual, max_dist = 1) |
          .contains_phrase(entry_description, .pat_accrual_unbilled),
        1L, 0L
      ),

      flag_internal_transfer = if_else(
        .contains_phrase(entry_description, .pat_internal_transfer),
        1L, 0L
      ),

      flag_bonus_payroll = if_else(
        .contains_approx_word(entry_description, .kw_bonus, max_dist = 1) |
          .contains_phrase(entry_description, .pat_payroll_tax),
        1L, 0L
      ),

      flag_credit_card = if_else(
        .contains_phrase(entry_description, .pat_credit_card),
        1L, 0L
      ),

      flag_valuation = if_else(
        .contains_approx_word(entry_description, .kw_valuation, max_dist = 1),
        1L, 0L
      ),

      flag_equity_shareholder = if_else(
        .contains_approx_word(entry_description, .kw_equity, max_dist = 1),
        1L, 0L
      ),

      flag_batch_processing = if_else(
        .contains_approx_word(entry_description, .kw_batch, max_dist = 1),
        1L, 0L
      )
    ) |>

    mutate(
      revenue_account_flag   = .make_account_flag(gl_account, revenue_accounts),
      accrual_account_flag   = .make_account_flag(gl_account, accrual_accounts),
      provision_account_flag = .make_account_flag(gl_account, provision_accounts)
    )
}


# ---- Custom recipe steps: leakage-safe group-statistic features ----
# Each step learns its lookup tables from the training fold in prep() and joins
# them on in bake(), filling unseen groups with the fallbacks noted at the top.


# ---- Timing: user_period_entry_count (entries per user per period) ----

step_timing_features <- function(recipe, role = NA, trained = FALSE,
                                 lookup_user_period = NULL,
                                 skip = FALSE, id = rand_id("timing_features")) {
  add_step(
    recipe,
    step_timing_features_new(
      role               = role,
      trained            = trained,
      lookup_user_period = lookup_user_period,
      skip               = skip,
      id                 = id
    )
  )
}

step_timing_features_new <- function(role, trained, lookup_user_period,
                                     skip, id) {
  step(
    subclass           = "timing_features",
    role               = role,
    trained            = trained,
    lookup_user_period = lookup_user_period,
    skip               = skip,
    id                 = id
  )
}

prep.step_timing_features <- function(x, training, info = NULL, ...) {

  lookup_user_period <- training |>
    group_by(company, user, gl_period) |>
    summarise(.up_count = n(), .groups = "drop")

  step_timing_features_new(
    role               = x$role,
    trained            = TRUE,
    lookup_user_period = lookup_user_period,
    skip               = x$skip,
    id                 = x$id
  )
}

bake.step_timing_features <- function(object, new_data, ...) {

  new_data <- new_data |>
    left_join(object$lookup_user_period, by = c("company", "user", "gl_period")) |>
    mutate(user_period_entry_count = replace_na(.up_count, 1L)) |>
    select(-.up_count)

  new_data
}

print.step_timing_features <- function(x, width = max(20, options()$width - 30), ...) {
  cat("Timing features (user-period entry count) [",
      if (x$trained) "trained" else "untrained", "]\n", sep = "")
  invisible(x)
}


# ---- Amount: journal_round_rate + roundness_relative_to_journal ----
# Needs is_round_1000 from add_static_features. Fallback: global round rate
# for unseen journals, 0 for the deviation.

step_amount_features <- function(recipe, role = NA, trained = FALSE,
                                 lookup_journal = NULL,
                                 global_round_rate = NULL,
                                 skip = FALSE, id = rand_id("amount_features")) {
  add_step(
    recipe,
    step_amount_features_new(
      role = role, trained = trained,
      lookup_journal    = lookup_journal,
      global_round_rate = global_round_rate,
      skip = skip, id = id
    )
  )
}

step_amount_features_new <- function(role, trained, lookup_journal,
                                     global_round_rate, skip, id) {
  step(
    subclass = "amount_features",
    role     = role,
    trained  = trained,
    lookup_journal    = lookup_journal,
    global_round_rate = global_round_rate,
    skip     = skip,
    id       = id
  )
}

prep.step_amount_features <- function(x, training, info = NULL, ...) {

  lookup_journal <- training |>
    group_by(company, journal) |>
    summarise(
      .j_round_rate = mean(is_round_1000, na.rm = TRUE),
      .groups       = "drop"
    )

  global_round_rate <- mean(training$is_round_1000, na.rm = TRUE)

  step_amount_features_new(
    role = x$role, trained = TRUE,
    lookup_journal    = lookup_journal,
    global_round_rate = global_round_rate,
    skip = x$skip, id = x$id
  )
}

bake.step_amount_features <- function(object, new_data, ...) {

  new_data <- new_data |>
    left_join(object$lookup_journal, by = c("company", "journal")) |>
    mutate(
      journal_round_rate = if_else(
        is.na(.j_round_rate),
        object$global_round_rate,
        .j_round_rate
      ),
      roundness_relative_to_journal = abs(is_round_1000 - journal_round_rate)
    ) |>
    select(-.j_round_rate)

  new_data
}

print.step_amount_features <- function(x, width = max(20, options()$width - 30), ...) {
  cat("Amount features (journal round rate, roundness relative to journal) [",
      if (x$trained) "trained" else "untrained", "]\n", sep = "")
  invisible(x)
}


# ---- User: user_account_rarity + user_account_period_entry_count ----
# Rarity = 1 - (user-account count / user total). Unseen groups: 1.0 for
# rarity, 1 for the count.

step_user_features <- function(recipe, role = NA, trained = FALSE,
                               lookup_user_account    = NULL,
                               lookup_user_acc_period = NULL,
                               skip = FALSE, id = rand_id("user_features")) {
  add_step(
    recipe,
    step_user_features_new(
      role = role, trained = trained,
      lookup_user_account    = lookup_user_account,
      lookup_user_acc_period = lookup_user_acc_period,
      skip = skip, id = id
    )
  )
}

step_user_features_new <- function(role, trained, lookup_user_account,
                                   lookup_user_acc_period, skip, id) {
  step(
    subclass = "user_features",
    role     = role,
    trained  = trained,
    lookup_user_account    = lookup_user_account,
    lookup_user_acc_period = lookup_user_acc_period,
    skip     = skip,
    id       = id
  )
}

prep.step_user_features <- function(x, training, info = NULL, ...) {

  user_totals <- training |>
    group_by(company, user) |>
    summarise(.user_total = n(), .groups = "drop")

  lookup_user_account <- training |>
    group_by(company, user, gl_account) |>
    summarise(.ua_count = n(), .groups = "drop") |>
    left_join(user_totals, by = c("company", "user")) |>
    mutate(.user_account_rarity = 1 - (.ua_count / .user_total)) |>
    select(company, user, gl_account, .user_account_rarity)

  lookup_user_acc_period <- training |>
    group_by(company, user, gl_account, gl_period) |>
    summarise(.uap_count = n(), .groups = "drop")

  step_user_features_new(
    role = x$role, trained = TRUE,
    lookup_user_account    = lookup_user_account,
    lookup_user_acc_period = lookup_user_acc_period,
    skip = x$skip, id = x$id
  )
}

bake.step_user_features <- function(object, new_data, ...) {

  new_data <- new_data |>

    left_join(object$lookup_user_account,
              by = c("company", "user", "gl_account")) |>
    mutate(user_account_rarity = replace_na(.user_account_rarity, 1)) |>
    select(-.user_account_rarity) |>

    left_join(object$lookup_user_acc_period,
              by = c("company", "user", "gl_account", "gl_period")) |>
    mutate(user_account_period_entry_count = replace_na(.uap_count, 1L)) |>
    select(-.uap_count)

  new_data
}

print.step_user_features <- function(x, width = max(20, options()$width - 30), ...) {
  cat("User features (account rarity, account-period entry count) [",
      if (x$trained) "trained" else "untrained", "]\n", sep = "")
  invisible(x)
}


# ---- Journal: journal_account_match_rarity + user_journal_rarity ----
# Both are 1 - (sub-group count / parent total). Unseen groups: 1.0.

step_journal_features <- function(recipe, role = NA, trained = FALSE,
                                  lookup_journal_account = NULL,
                                  lookup_user_journal    = NULL,
                                  skip = FALSE,
                                  id = rand_id("journal_features")) {
  add_step(
    recipe,
    step_journal_features_new(
      role = role, trained = trained,
      lookup_journal_account = lookup_journal_account,
      lookup_user_journal    = lookup_user_journal,
      skip = skip,
      id   = id
    )
  )
}

step_journal_features_new <- function(role, trained, lookup_journal_account,
                                      lookup_user_journal, skip, id) {
  step(
    subclass = "journal_features",
    role     = role,
    trained  = trained,
    lookup_journal_account = lookup_journal_account,
    lookup_user_journal    = lookup_user_journal,
    skip     = skip,
    id       = id
  )
}

prep.step_journal_features <- function(x, training, info = NULL, ...) {

  journal_totals <- training |>
    group_by(company, journal) |>
    summarise(.j_total = n(), .groups = "drop")

  lookup_journal_account <- training |>
    group_by(company, journal, gl_account) |>
    summarise(.ja_count = n(), .groups = "drop") |>
    left_join(journal_totals, by = c("company", "journal")) |>
    mutate(.journal_account_match_rarity = 1 - (.ja_count / .j_total)) |>
    select(company, journal, gl_account, .journal_account_match_rarity)

  user_totals <- training |>
    group_by(company, user) |>
    summarise(.user_total = n(), .groups = "drop")

  lookup_user_journal <- training |>
    group_by(company, user, journal) |>
    summarise(.uj_count = n(), .groups = "drop") |>
    left_join(user_totals, by = c("company", "user")) |>
    mutate(.user_journal_rarity = 1 - (.uj_count / .user_total)) |>
    select(company, user, journal, .user_journal_rarity)

  step_journal_features_new(
    role = x$role, trained = TRUE,
    lookup_journal_account = lookup_journal_account,
    lookup_user_journal    = lookup_user_journal,
    skip = x$skip, id = x$id
  )
}

bake.step_journal_features <- function(object, new_data, ...) {

  new_data <- new_data |>

    left_join(object$lookup_journal_account,
              by = c("company", "journal", "gl_account")) |>
    mutate(
      journal_account_match_rarity = replace_na(.journal_account_match_rarity, 1)
    ) |>
    select(-.journal_account_match_rarity) |>

    left_join(object$lookup_user_journal,
              by = c("company", "user", "journal")) |>
    mutate(
      user_journal_rarity = replace_na(.user_journal_rarity, 1)
    ) |>
    select(-.user_journal_rarity)

  new_data
}

print.step_journal_features <- function(x, width = max(20, options()$width - 30), ...) {
  cat("Journal features (journal-account rarity, user-journal rarity) [",
      if (x$trained) "trained" else "untrained", "]\n", sep = "")
  invisible(x)
}


# ---- Per-company centering and scaling ----
# Replaces global step_normalize() for the few features whose raw scale depends
# on company size. Centers/scales each value with its own company's training
# mean and SD; unseen companies or zero-SD columns fall back to global stats.

step_company_normalize <- function(recipe, ...,
                                   role = NA, trained = FALSE,
                                   columns = NULL,
                                   lookup = NULL,
                                   global_stats = NULL,
                                   skip = FALSE,
                                   id = rand_id("company_normalize")) {
  terms <- enquos(...)
  add_step(
    recipe,
    step_company_normalize_new(
      terms        = terms,
      role         = role,
      trained      = trained,
      columns      = columns,
      lookup       = lookup,
      global_stats = global_stats,
      skip         = skip,
      id           = id
    )
  )
}

step_company_normalize_new <- function(terms, role, trained, columns,
                                       lookup, global_stats, skip, id) {
  step(
    subclass     = "company_normalize",
    terms        = terms,
    role         = role,
    trained      = trained,
    columns      = columns,
    lookup       = lookup,
    global_stats = global_stats,
    skip         = skip,
    id           = id
  )
}

prep.step_company_normalize <- function(x, training, info = NULL, ...) {

  col_names <- recipes_eval_select(x$terms, training, info)

  lookup <- training |>
    group_by(company) |>
    summarise(
      across(all_of(col_names),
             list(mean = ~mean(.x, na.rm = TRUE),
                  sd   = ~sd(.x,   na.rm = TRUE)),
             .names = "{.col}__{.fn}"),
      .groups = "drop"
    )

  global_stats <- training |>
    summarise(
      across(all_of(col_names),
             list(mean = ~mean(.x, na.rm = TRUE),
                  sd   = ~sd(.x,   na.rm = TRUE)),
             .names = "{.col}__{.fn}")
    )

  step_company_normalize_new(
    terms        = x$terms,
    role         = x$role,
    trained      = TRUE,
    columns      = col_names,
    lookup       = lookup,
    global_stats = global_stats,
    skip         = x$skip,
    id           = x$id
  )
}

bake.step_company_normalize <- function(object, new_data, ...) {

  new_data <- new_data |>
    left_join(object$lookup, by = "company")

  for (col in object$columns) {
    mean_col <- paste0(col, "__mean")
    sd_col   <- paste0(col, "__sd")

    global_mean <- object$global_stats[[mean_col]]
    global_sd   <- object$global_stats[[sd_col]]

    eff_mean <- dplyr::coalesce(new_data[[mean_col]], global_mean)
    eff_sd   <- dplyr::coalesce(new_data[[sd_col]],   global_sd)

    # Guard against zero/NA SD so we never divide by zero (center only)
    eff_sd <- dplyr::if_else(
      is.na(eff_sd) | eff_sd == 0,
      dplyr::if_else(is.na(global_sd) | global_sd == 0, 1, global_sd),
      eff_sd
    )

    new_data[[col]] <- (new_data[[col]] - eff_mean) / eff_sd
  }

  drop_cols <- c(
    paste0(object$columns, "__mean"),
    paste0(object$columns, "__sd")
  )
  new_data <- new_data |> select(-all_of(drop_cols))

  new_data
}

print.step_company_normalize <- function(x, width = max(20, options()$width - 30), ...) {
  cat("Per-company normalization [",
      if (x$trained) "trained" else "untrained", "]\n", sep = "")
  invisible(x)
}
