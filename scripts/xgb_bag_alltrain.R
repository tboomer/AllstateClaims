# Run XGBoost model on full train data set to create oob values for bagging.

library(readr)
library(caret)
library(dplyr)
library(xgboost)
source('./scripts/bagger.R')

xg_eval_mae <- function (yhat, dtrain) {
    y = getinfo(dtrain, "label")
    err= calc_mae(exp(y),exp(yhat) )
    return (list(metric = "error", value = err))
}

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

# Read data
train <- read_csv('./source/train.csv.zip')
test <- read_csv('./source/test.csv.zip')

# Transform character to factor variables with common levels across 
# train/test
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

# Call bagger script
oob_pred <- train_bags(train, "logloss", 10)

save(oob_pred, file = './cache/xgb_oob_1,RData')
# 
# features=names(train)
# 
# 
# dtrain <- xgb.DMatrix(data=data.matrix(train[,2:131]),
#                       label=data.matrix(train$logloss))
# watchlist<-list(dtrain = dtrain)
# 
# Sys.time()
# set.seed(575)
# clf <- xgb.train(params=list(  objective="reg:linear", 
#                                booster = "gbtree",
#                                eta=0.1, 
#                                max_depth=7, 
#                                subsample=0.85,
#                                colsample_bytree=0.7) ,
#                  data = dtrain, 
#                  nrounds = 200, 
#                  verbose = 1,
#                  print_every_n=5,
#                  early_stopping_rounds    = 15,
#                  watchlist           = watchlist,
#                  maximize            = FALSE,
#                  feval = xg_eval_mae
# )
# Sys.time()


#-------------------------------------------------------------------------------
# Make prediction on test data
# test[, 2:117] <- lapply(test[, 2:117], as.factor)
# testpred <- predict(clf, xgb.DMatrix(data=data.matrix(test[,2:131])))
# testpred <- exp(testpred) - 1
# 
# submission <- data.frame(id = test$id, loss = testpred)
# write_csv(submission, './submissions/compare1102B.csv')
                                               