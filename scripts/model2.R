
library(readr)
library(caret)
library(dplyr)
library(xgboost)

xg_eval_mae <- function (yhat, dtrain) {
    y = getinfo(dtrain, "label")
    err= calc_mae(exp(y),exp(yhat) )
    return (list(metric = "error", value = err))
}


calc_mae <- function(preds, y) {
    sum(abs(y - preds))/length(y)
}

train <- read_csv('./source/train.csv.zip')
test <- read_csv('./source/test.csv.zip')

train$logloss <- log(train$loss + 1)
train[, 2:117] <- lapply(train[, 2:117], as.factor)

features=names(train)

grid <- list(depth = c(7,8,9),
             subsamp = c(.8, .85, .9),
             colsamp = c(.4, .5, .6))
index <- matrix(c(rep(1,9),rep(2,9), rep(3,9),
                     rep(c(rep(1,3), rep(2,3), rep(3,3)),3),
                     rep(1:3, 9)), nrow = 27, ncol = 3)
result <- matrix(0, nrow = 27, ncol = 4, 
                 dimnames = list(NULL, c("depth", "subsamp", "colsamp", "mae")))



dtrain <- xgb.DMatrix(data=data.matrix(train[,2:131]),
                      label=data.matrix(train$logloss))
watchlist <- list(train = dtrain)
Sys.time()
set.seed(575)

for(i in 1:27) {
    xgb_params <- list(  objective="reg:linear", 
                         booster = "gbtree",
                         eta=0.1, 
                         max_depth=grid$depth[index[i,1]], 
                         subsample=grid$subsamp[index[i,2]],
                         colsample_bytree=grid$colsamp[index[i,3]])
    
    clf_cv <- xgb.cv(params = xgb_params,
                     data = dtrain,
                     nfold = 5,
                     nrounds = 200, 
                     print_every_n = 5,
                     verbose = TRUE,
                     watchlist = watchlist,
                     early_stopping_rounds    = 15,
                     maximize            = FALSE,
                     feval = xg_eval_mae)
    
    result[i,] <- c(xgb_params$max_depth, xgb_params$subsample, 
                    xgb_params$colsample_bytree, min(clf_cv$test.error.mean))
    
                    print(result[i,])
                    print(Sys.time())
}



# Predict on validation set
valpred <- predict(clf, dval)
valpred <- exp(valpred) - 1
calc_mae(myval$loss, valpred)

errordata <- data.frame(valpred, myval$loss)
ggplot(errordata, aes(myval$loss, valpred)) + geom_point() + 
    geom_abline(slope = 1, color = "blue")

#-------------------------------------------------------------------------------
# Make prediction on test data
test[, 2:117] <- lapply(test[, 2:117], as.factor)
testpred <- predict(clf, xgb.DMatrix(data=data.matrix(test[,2:131])))
testpred <- exp(testpred) - 1

submission <- data.frame(id = test$id, loss = testpred)
write_csv(submission, './submissions/submission1014A.csv')
                                               