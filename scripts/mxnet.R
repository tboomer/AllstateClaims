# Neural net model using train/validation set split to diagnose model performance.

library(readr)
library(caret)
library(dplyr)
library(mxnet)

# Define cost functions
# Cost function in XGBoost format 
xg_eval_mae <- function (yhat, dtrain) {
    y = getinfo(dtrain, "label")
    err= calc_mae(exp(y),exp(yhat) )
    return (list(metric = "error", value = err))
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
rm(all)


# Log-transform loss variable
train$logloss <- log(train$loss + 1)

features=names(train)

# Split train and validation sets
set.seed(36)
index <- createDataPartition(train$loss, p=.85, list = FALSE)
mytrain <- train[index,]
myval <- train[-index,]
rm(index)

# Define nnet configuration
data <- mx.symbol.Variable("data")


# Train XGBoost model
set.seed(56)
xgb_val <- sample_n(mytrain, 3000) 

xtrain <- data.matrix(mytrain[,2:131])
ytrain <- data.matrix(mytrain$logloss)


Sys.time()
set.seed(575)
clf <- xgb.train(params=list(  objective="reg:linear", 
                               booster = "gbtree",
                               eta=0.1, 
                               max_depth=7, 
                               subsample=0.85,
                               colsample_bytree=0.7) ,
                 data = dtrain, 
                 nrounds = 200, 
                 verbose = 1,
                 print_every_n=5,
                 early_stopping_rounds    = 15,
                 watchlist           = watchlist,
                 maximize            = FALSE,
                 # eval = "RMSE"
                 feval = xg_eval_mae
)
Sys.time()

# Predict on validation set
valpred <- predict(clf, data.matrix(myval[,2:131]))
valpred <- exp(valpred) - 1
calc_mae(myval$loss, valpred)

#-------------------------------------------------------------------------------
# DIAGNOSTICS

# Plot prediction vs actual
errordata <- cbind(myval, valpred, myval$loss)
ggplot(errordata, aes(myval$loss, valpred)) + geom_point() + 
    geom_abline(slope = 1, color = "blue")

# Print variable importance.
importance <- xgb.importance(feature_names = features, model = clf)
print.data.frame(importance)

