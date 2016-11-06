# Run XGBoost inside caret wrapper using 5-fold CV
library(readr)
library(caret)
library(plyr)
library(dplyr)
library(neuralnet)
library(Metrics)

# Define cost functions
# Custom MAE metric in caret format
mae_metric <- function (data,
                        lev = NULL,
                        model = NULL) {
    out <- mae(exp(data$obs)-1, exp(data$pred)-1)  
    names(out) <- "MAE"
    out
}


# Calculate mean average error
calc_mae <- function(preds, y) {
    sum(abs(y - preds))/length(y)
}

# Function to assign a new factor level to factors that appear <= n times.
reassign_levels <- function(var, n, new_val = "XX") {
    new_levels <- names(table(var))
    new_levels[table(var) <= n] <- new_val
    levels(var) <- new_levels
    return(var)
}

# Read source data
train <- read_csv('./source/train.csv.zip')
test <- read_csv('./source/test.csv.zip')

# Transform character to factor variables with common train/test levels
test$loss <- -99
all <- rbind(train, test)
all[, 2:117] <- lapply(all[, 2:117], as.factor)

# Consolidate factor levels with fewer than n instances. Apply this logic only
# to factor variables with >= p levels.
n <- 2
p <- 20
num_levels <- sapply(all[,2:117], function(x) length(levels(x)))
factor_names <- names(all[,2:117])
col_index <- factor_names[num_levels >= p]
all[, col_index] <- lapply(all[, col_index], function(x) reassign_levels(x,n))

train <- filter(all, loss != -99)
test <- filter(all, loss == -99)
test <- select(test, -loss)
rm(all)


# Log-transform loss variable
train$logloss <- log(train$loss + 1)

train_nn <- select(train, -loss, -id)
# index <- sample(nrow(train), 5000) # Subsample training set
train_nn <- train_nn[index,]


ctrl <- trainControl(method = "cv", 
                     number = 5,
                     summaryFunction = mae_metric,
                     verboseIter = TRUE,
                     allowParallel = TRUE)

tune <- data.frame(layer1 = 60,
                   layer2 = 20,
                   layer3 = 0)

set.seed(56)
nn_model <- train(logloss ~ .,
                  data = train_nn,
                  method = "neuralnet",
                  trControl = ctrl,
                  tuneGrid = tune,
                  metric = "MAE",
                  tuneLength = 1)
#-------------------------------------------------------------------------------
# Make prediction on test data
testpred <- predict(xgb_model, data.matrix(test[,2:131]))
testpred <- exp(testpred) - 1

submission <- data.frame(id = test$id, loss = testpred)
write_csv(submission, './submissions/compare1102C.csv')