library(skimr)
library(tidyverse)
library(tidymodels)
library(dplyr)
library(tibble)
library(digest)
library(readxl)
library(GGally)
library(ggplot2)
library(arrow)


# Data cleaning: consolidate the four company general ledgers into a single
# dataset. Loads the raw journal exports and GL master data, harmonises field
# names and types, and writes the merged parquet used by the rest of the pipeline.


### Import journal data

company_a_raw <- read_parquet("Company_A_Final.parquet")

company_b_raw <- read_parquet("Company_B_Final.parquet")

company_c_raw <- read_parquet("Company_C_Final.parquet")

company_d_raw <- read_parquet("Company_D_Final.parquet")


### Import GL master data

gl_master_a <- read_parquet("GL_MASTER_DATA_Company_A.parquet")

gl_master_b <- read_parquet("GL_MASTER_DATA_Company_B.parquet")

# Company C master data has duplicated rows and trailing whitespace to fix
gl_master_c <- read_parquet("GL_MASTER_DATA_Company_C.parquet")
gl_master_c$`%RGS` <- NA
gl_master_c <- gl_master_c %>%
  mutate(
    Grootboekrekeningnaam = str_trim(Grootboekrekeningnaam, side = "right"),
    Grootboekfilter = str_trim(Grootboekfilter, side = "right")
  )
gl_master_c <- gl_master_c %>%
  mutate(
    Grootboekrekeningnaam = replace(Grootboekrekeningnaam, 196, NA),
    Grootboekfilter = replace(Grootboekfilter, 196, NA)
  )

gl_master_d <- read_parquet("GL_MASTER_DATA_Company_D.parquet")


### Add GL account name and type from the master data

company_a_enrich <- company_a_raw |>
  left_join(distinct(gl_master_a) |>
              select(`%GL_ACC`, Grootboekrekeningnaam, Grootboekrekeningtype),
            by = "%GL_ACC")

company_b_enrich <- company_b_raw |>
  left_join(distinct(gl_master_b) |>
              select(`%GL_ACC`, Grootboekrekeningnaam, Grootboekrekeningtype),
            by = "%GL_ACC")

company_c_enrich <- company_c_raw |>
  left_join(distinct(gl_master_c) |>
              select(`%GL_ACC`, Grootboekrekeningnaam, Grootboekrekeningtype),
            by = "%GL_ACC")

company_d_enrich <- company_d_raw |>
  left_join(gl_master_d |>
              select(`%GL_ACC`, Grootboekrekeningnaam, Grootboekrekeningtype),
            by = "%GL_ACC")


### Rename and keep only the relevant fields

company_a_clean <- company_a_enrich |>
  rename(
    index_nr            = `%GL_LINE`,
    risk_score          = risk_score,
    risk_reason         = risk_reason,
    company             = Bedrijfsnaam,
    journal_entry_number= Boekstuknummer,
    journal             = Dagboek,
    user                = Gebruiker,
    relation_type       = Relatietype,
    relation_name       = Relatienaam,
    document            = `%GL_DOC`,
    gl_period           = Periode,
    post_date           = Boekdatum,
    entry_date          = Invoerdatum,
    gl_account          = `%GL_ACC`,
    gl_description      = Grootboekrekeningnaam,
    gl_type             = Grootboekrekeningtype,
    entry_description   = `Mutatie omschrijving`,
    amount              = `#Bedrag`,
    cost_center         = Kostenplaats
  ) |>
  select(
    index_nr,
    risk_score,
    risk_reason,
    company,
    journal_entry_number,
    journal,
    user,
    relation_type,
    relation_name,
    document,
    gl_period,
    post_date,
    entry_date,
    gl_account,
    gl_description,
    gl_type,
    entry_description,
    amount,
    cost_center
  )

company_b_clean <- company_b_enrich |>
  rename(
    index_nr            = `%GL_LINE`,
    risk_score          = risk_score,
    risk_reason         = risk_reason,
    company             = Bedrijfsnaam,
    journal_entry_number= Boekstuknummer,
    journal             = Dagboek,
    user                = Gebruiker,
    relation_type       = Relatietype,
    relation_name       = Relatienaam,
    document            = `%GL_DOC`,
    gl_period           = Periode,
    post_date           = Boekdatum,
    entry_date          = Invoerdatum,
    gl_account          = `%GL_ACC`,
    gl_description      = Grootboekrekeningnaam,
    gl_type             = Grootboekrekeningtype,
    entry_description   = `Mutatie omschrijving`,
    amount              = `#Bedrag`,
    cost_center         = Kostenplaats
  ) |>
  select(
    index_nr,
    risk_score,
    risk_reason,
    company,
    journal_entry_number,
    journal,
    user,
    relation_type,
    relation_name,
    document,
    gl_period,
    post_date,
    entry_date,
    gl_account,
    gl_description,
    gl_type,
    entry_description,
    amount,
    cost_center
  )

# Company C posts periods as German month names; map them back to numbers
company_c_clean <- company_c_enrich |>
  rename(
    index_nr             = `%GL_LINE`,
    risk_score           = risk_score,
    risk_reason          = risk_reason,
    company              = Bedrijfsnaam,
    journal_entry_number = Boekstuknummer,
    journal              = Dagboek,
    user                 = Gebruiker,
    relation_type        = Relatietype,
    relation_name        = Relatienaam,
    document             = `%GL_DOC`,
    gl_period            = Periode,
    post_date            = Boekdatum,
    entry_date           = Invoerdatum,
    gl_account           = `%GL_ACC`,
    gl_description       = Grootboekrekeningnaam,
    gl_type              = Grootboekrekeningtype,
    entry_description    = `Mutatie omschrijving`,
    amount               = `#Bedrag`,
    cost_center          = Kostenplaats
  ) |>
  select(
    index_nr,
    risk_score,
    risk_reason,
    company,
    journal_entry_number,
    journal,
    user,
    relation_type,
    relation_name,
    document,
    gl_period,
    post_date,
    entry_date,
    gl_account,
    gl_description,
    gl_type,
    entry_description,
    amount,
    cost_center
  ) |>
  mutate(
    gl_period = trimws(as.character(gl_period)),
    gl_period = case_when(
      gl_period %in% c("Januar", "januari") ~ 1,
      gl_period == "Februar" ~ 2,
      gl_period == "März" ~ 3,
      gl_period == "April" ~ 4,
      gl_period == "Mai" ~ 5,
      gl_period == "Juni" ~ 6,
      gl_period == "Juli" ~ 7,
      gl_period == "August" ~ 8,
      gl_period == "September" ~ 9,
      gl_period == "Oktober" ~ 10,
      gl_period == "November" ~ 11,
      gl_period == "Dezember" ~ 12,
      TRUE ~ NA_real_
    )
  )

company_d_clean <- company_d_enrich |>
  rename(
    index_nr            = `%GL_LINE`,
    risk_score          = risk_score,
    risk_reason         = risk_reason,
    company             = Bedrijfsnaam,
    journal_entry_number= Boekstuknummer,
    journal             = Dagboek,
    user                = Gebruiker,
    relation_type       = Relatietype,
    relation_name       = Relatienaam,
    document            = `%GL_DOC`,
    gl_period           = Periode,
    post_date           = Boekdatum,
    entry_date          = Invoerdatum,
    gl_account          = `%GL_ACC`,
    gl_description      = Grootboekrekeningnaam,
    gl_type             = Grootboekrekeningtype,
    entry_description   = `Mutatie omschrijving`,
    amount              = `#Bedrag`,
    cost_center         = Kostenplaats
  ) |>
  select(
    index_nr,
    risk_score,
    risk_reason,
    company,
    journal_entry_number,
    journal,
    user,
    relation_type,
    relation_name,
    document,
    gl_period,
    post_date,
    entry_date,
    gl_account,
    gl_description,
    gl_type,
    entry_description,
    amount,
    cost_center
  )


### Standardize column types

standardize_types <- function(df) {

  df %>%
    mutate(
      index_nr             = as.double(index_nr),
      company              = as.character(company),
      risk_score           = as.double(risk_score),
      risk_reason          = as.character(risk_reason),
      amount               = as.double(amount),
      gl_account           = as.character(gl_account),
      gl_description       = as.character(gl_description),
      gl_type              = as.character(gl_type),
      journal              = as.character(journal),
      entry_description    = as.character(entry_description),
      journal_entry_number = as.double(journal_entry_number),
      post_date            = as.POSIXct(post_date),
      entry_date           = as.POSIXct(entry_date),
      gl_period            = as.double(gl_period),
      user                 = as.character(user),
      relation_name        = as.character(relation_name),
      relation_type        = as.character(relation_type),
      document             = as.character(document),
      cost_center          = as.character(cost_center)
    )
}

company_a_clean <- standardize_types(company_a_clean)
company_b_clean <- standardize_types(company_b_clean)
company_c_clean <- standardize_types(company_c_clean)
company_d_clean <- standardize_types(company_d_clean)


### Merge into one dataset

company_a_clean <- company_a_clean |> mutate(source_dataset = "Company_A")
company_b_clean <- company_b_clean |> mutate(source_dataset = "Company_B")
company_c_clean <- company_c_clean |> mutate(source_dataset = "Company_C")
company_d_clean <- company_d_clean |> mutate(source_dataset = "Company_D")

final_db <- bind_rows(
  company_d_clean,
  company_c_clean,
  company_b_clean,
  company_a_clean
)

# index_nr is only unique within a company, so combine it with source_dataset
# for a globally unique key used by every downstream join.
final_db <- final_db |>
  mutate(unique_id = paste0(source_dataset, "_", index_nr))

n_total  <- nrow(final_db)
n_unique <- n_distinct(final_db$unique_id)

cat("Total rows:", n_total, "\n")
cat("Unique unique_id values:", n_unique, "\n")
cat("Duplicates:", n_total - n_unique, "\n")

stopifnot(n_total == n_unique)


### Export merged dataset

write_parquet(final_db, "~/Desktop/BAM - RSM/Thesis/Work_Material/0. Data/GL_Dataset.parquet")
