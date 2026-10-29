# =============================================================================
# create_borrow_repay_csv.R
#
# PURPOSE: 
#    Prepare the borrow_repay survival dataset from FinSurvival raw RDS
#    and export a cleaned CSV ready for Python preprocessing in Data_Preprocessing.
#
# HOW TO RUN:
#   This script must be run from the FinSurvival repository:
#     https://github.com/Large-Transaction-Models/DMLR_DeFi_Survival_Benchmark
#   The raw data files are expected there at ./Data/Survival_Data/Borrow/
#
# OUTPUT:
#   Writes borrow_repay_first_obs.csv to the current working directory.
#   After running, move the file to:
#     'Data_Preprocessing/data/Finsurvival/borrow_repay/borrow_repay_first_obs.csv'
#   This file is then used in run_preprocess_borrow_repay.py 
# =============================================================================

library(readr)
library(dplyr)
library(stringr)

not_all_na <- function(x) any(!is.na(x))

cat("=== Loading Borrow-Repay data ===\n")
dataPath <- "./Data/Survival_Data/"

load_shards <- function(path) {
  files <- list.files(path, full.names = TRUE)
  df <- data.frame()
  for (f in files) {
    df <- bind_rows(df, read_rds(f)) %>%
      select(where(not_all_na)) %>%
      select(-starts_with("exo")) %>%
      filter(!is.na(id))
  }
  df
}

X_train <- load_shards(paste0(dataPath, "Borrow/X_train/"))
X_test  <- load_shards(paste0(dataPath, "Borrow/X_test/"))

y_train <- read_rds(paste0(dataPath, "Borrow/Repay/y_train.rds")) %>% filter(!is.na(id))
y_test  <- read_rds(paste0(dataPath, "Borrow/Repay/y_test.rds"))  %>% filter(!is.na(id))

train <- inner_join(y_train, X_train, by = "id", relationship = "many-to-many")
test  <- inner_join(y_test, X_test, by = "id", relationship = "many-to-many")

shared_cols <- intersect(names(train), names(test))
full_data <- bind_rows(
  train %>% select(all_of(shared_cols)),
  test  %>% select(all_of(shared_cols))
)
cat("Full data:", nrow(full_data), "rows,", ncol(full_data), "cols\n")

# only use first observation per user 
cat("Restricting to first observation per user:n")
first_obs <- full_data %>%
  group_by(user) %>%
  arrange(timestamp) %>%
  slice(1) %>%
  ungroup()
cat("After restriction:", nrow(first_obs), "rows\n")

# Filter timeDiff > 0
first_obs <- first_obs %>% filter(timeDiff > 0)
cat("After timeDiff > 0 filter:", nrow(first_obs), "rows\n")

# Drop degenerate user-history features (
cols_to_drop_degenerate <- c(
  "userCollateralCount",
  "userDepositCount", "userDepositSum", "userDepositSumUSD", "userDepositSumETH",
  "userWithdrawCount", "userWithdrawSum", "userWithdrawAvgAmount",
  "userWithdrawSumUSD", "userWithdrawAvgAmountUSD", "userWithdrawSumETH", "userWithdrawAvgAmountETH",
  "userBorrowCount", "userBorrowSum", "userBorrowSumUSD", "userBorrowSumETH",
  "userRepayCount", "userRepaySum", "userRepayAvgAmount",
  "userRepaySumUSD", "userRepayAvgAmountUSD", "userRepaySumETH", "userRepayAvgAmountETH",
  "userSwapCount",
  "userLiquidationCount", "userLiquidationSum", "userLiquidationAvgAmount",
  "userLiquidationSumUSD", "userLiquidationAvgAmountUSD",
  "userLiquidationSumETH", "userLiquidationAvgAmountETH",
  "userActiveDaysWeekly", "userActiveDaysMonthly", "userActiveDaysYearly",
  "userSecondsSinceFirstTransaction",
  "userDepositAvgAmount", "userDepositAvgAmountUSD", "userDepositAvgAmountETH",
  "userIsNew"
)

# Drop metadata/identifier columns 
cols_to_drop_meta <- c("id", "user", "timestamp", "type", "pool",
                       "Index Event", "Outcome Event",
                       "timeDiff", "status")

all_drops <- unique(c(cols_to_drop_degenerate, cols_to_drop_meta))
df <- first_obs %>% select(-any_of(all_drops))

#  Create target columns 
df$time  <- first_obs$timeDiff / 86400
df$event <- first_obs$status

cat("\n\tRenaming columns with prefixes..n")

numeric_cols <- names(df)[sapply(df, is.numeric)]
numeric_cols <- setdiff(numeric_cols, c("time", "event"))

categorical_cols <- names(df)[sapply(df, function(x) is.character(x) | is.factor(x))]

rename_map <- setNames(
  c(numeric_cols, categorical_cols),
  c(paste0("num_", numeric_cols), paste0("fac_", categorical_cols))
)
df <- df %>% rename(!!!rename_map)

cat("Final dataset:", nrow(df), "rows,", ncol(df), "cols\n")
cat("Numeric features:", sum(startsWith(names(df), "num_")), "\n")
cat("Categorical features:", sum(startsWith(names(df), "fac_")), "\n")
cat("Target columns: time, event\n\n")

cat("Columns:\n")
cat(paste(names(df), collapse = "\n"), "\n\n")

cat("Event distribution:\n")
print(table(df$event, useNA = "always"))
cat("\nTime (days) summary:\n")
print(summary(df$time))

# write csv (do not forget to move!)
out_path <- "borrow_repay_first_obs.csv"
write.csv(df, file = out_path, row.names = FALSE)
cat("\nWritten to:", out_path, "\n")
cat("File size:", round(file.size(out_path) / 1024 / 1024, 1), "MB\n")
