# XGBoost model using train/validation set split to diagnose model performance.

library(readr)
library(caret)
library(dplyr)
library(xgboost)

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

# Read source data
train <- read_csv('./source/train.csv.zip')
test <- read_csv('./source/test.csv.zip')

# Modify features
train$logloss <- log(train$loss + 1)
train[, 2:117] <- lapply(train[, 2:117], as.factor)

features=names(train)

# Split train and validation sets
set.seed(36)
index <- createDataPartition(train$loss, p=.85, list = FALSE)
mytrain <- train[index,]
myval <- train[-index,]
rm(index)

mytrain <- filter(mytrain, loss < 30000)

# Train XGBoost model
set.seed(56)
xgb_val <- sample_n(train, 3000) 

dtrain <- xgb.DMatrix(data=data.matrix(mytrain[,2:131]),
                      label=data.matrix(mytrain$logloss))
dval <- xgb.DMatrix(data=data.matrix(xgb_val[,2:131]),
                  label=data.matrix(xgb_val$logloss))
watchlist<-list(dval=dval, dtrain=dtrain)

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

