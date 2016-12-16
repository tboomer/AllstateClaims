# Run XGBoost model on full train data set to create oob values for bagging.

library(readr)
library(caret)
library(dplyr)


# Function to assign a new factor level to factors that appear <= n times.
reassign_levels <- function(var, n, new_val = "XX") {
    new_levels <- names(table(var))
    new_levels[table(var) <= n] <- new_val
    levels(var) <- new_levels
    return(var)
}

# Read data
train <- read_csv('./source/train.csv.zip')
test <- read_csv('./source/test.csv.zip')

# Log-transform loss variable
train$logloss <- log(train$loss)

# Transform character to factor variables with common levels across 
# train/test
test$loss <- -1
test$logloss <- NA

all <- rbind(train, test)
all[, 2:117] <- lapply(all[, 2:117], as.factor)



# Consolidate factor levels with fewer than n instances. Apply this logic only
# to factor variables with >= p levels.
n <- 6
p <- 20
num_levels <- sapply(all[,2:117], function(x) length(levels(x)))
factor_names <- names(all[,2:117])
col_index <- factor_names[num_levels >= p]
all[, col_index] <- lapply(all[, col_index], function(x) reassign_levels(x,n))

train <- filter(all, loss != -1)
test <- filter(all, loss == -1)
test <- select(test, -loss, -logloss)
rm(all)

# # Remap low frequency categories to NIT, HII, LOW
# 
#     cat_values <- group_by(all, cat116) %>% 
#         summarize(count = n(), avg_loss = mean(loss))
#     
#     for(i in 1:nrow(cat_values)) {
#         if(cat_values$count[i] > 3) {
#             next
#         } else if(cat_values$avg_loss[i] == -1)  {
#             cat_values[i,1] = "NIT"
#         } else if(cat_values$avg_loss[i] < 10000) {
#             cat_values[i,1] = "LOW"
#         } else cat_values[i,1] = "HII"
#     }
#     df[, col_num] <- as.factor(df[, col_num])

