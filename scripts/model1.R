
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

set.seed(360)
index <- createDataPartition(train$loss, p=.85, list = FALSE)
mytrain <- train[index,]
myval <- train[-index,]
rm(index)


features=names(train)


set.seed(56)
wltst=sample(nrow(mytrain),3000)  

dtrain <- xgb.DMatrix(data=data.matrix(mytrain[,2:131]),
                      label=data.matrix(mytrain$logloss))
dval <- xgb.DMatrix(data=data.matrix(myval[,2:131]),
                  label=data.matrix(myval$logloss))
watchlist<-list(dval=dval, dtrain=dtrain)

Sys.time()
set.seed(575)
clf <- xgb.train(params=list(  objective="reg:linear", 
                               booster = "gbtree",
                               eta=0.1, 
                               max_depth=6, 
                               subsample=0.85,
                               colsample_bytree=0.7) ,
                 data = dtrain, 
                 nrounds = 100, 
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
                                               