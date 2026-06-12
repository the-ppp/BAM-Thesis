# BAM-Thesis

Code for my master's thesis on transaction-level risk scoring of journal entries
in an auditing context. The pipeline cleans general ledger data from four
companies (anonymised as Company A–D), engineers audit-relevant features, trains
five models under three class-imbalance strategies, and evaluates them with
ranking metrics and a per-transaction explanation layer.

## Pipeline

Run the scripts in folder order; each stage writes outputs that the next reads.

1. **Data Cleaning** — `Data_Cleaning.R` merges the four company ledgers into one dataset.
2. **Feature Engineering** — `Feature_Engineering.R` defines the feature and recipe steps (sourced by the modeling scripts).
3. **Data Modeling** — `Modeling_simple.R` (logistic regression, elastic net, decision tree), then `Modeling_complex.R` (random forest, XGBoost). `Metrics.R` holds the custom ranking metrics.
4. **Evaluation** — `Evaluation.R`, then `Explanation_Interpretability.R`, `Feature_Importance_Analysis.R`, and `Visualizations.R`.

## Data

The raw general ledger data is confidential client data and is not included in
this repository. Client names are also masked throughout the code, with the four
companies referred to only as Company A–D.

## Requirements

R with tidymodels, themis, ranger, xgboost, glmnet, treeshap, and arrow.
