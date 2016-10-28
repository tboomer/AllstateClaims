
library(readr)
library(caret)
library(dplyr)
library(doParallel)
library(xgboost)

xg_eval_mae <- function (yhat, dtrain) {
    y = getinfo(dtrain, "label")
    err= mae(exp(y),exp(yhat) )
    return (list(metric = "error", value = err))
}

calc_mae <- function(preds, y) {
    sum(abs(y - preds))/length(y)
}
# data is a matrix of predictions and observations
caret_mae <- function(data) {
    mae <- sum(abs(data$obs - data$pred))/nrow(data)
    names(mae) <- "MAE"
    return(mae)
}

train <- read_csv('./source/train.csv.zip')
test <- read_csv('./source/test.csv.zip')

train$logloss <- log(train$loss + 1)
train[, 2:117] <- lapply(train[, 2:117], as.factor)

trainX <- data.matrix(select(train, -loss, -id))
trainY <- train$logloss

ctrl <- trainControl(method = "cv", 
                     number = 5,						
                     # summaryFunction = "RMSE",
                     nrounds = 50,
                     allowParallel = TRUE)
grid <- expand.grid(interaction.depth=c(1,2), # Depth of variable interactions
                    n.trees=c(10,20),	        # Num trees to fit
                    shrinkage=c(0.01,0.1),		# Try 2 values for learning rate 
                    n.minobsinnode = 20)

cl <- 4
registerDoParallel(cl) 
getDoParWorkers()

set.seed(56)
xgb_model <- train(x = trainX, y = trainY,
                  method = "xgbLinear",
                  # metric = "RMSE",
                  trControl = ctrl,
                  # tuneGrid=grid,
                  verbose=FALSE)

stopCluster(cl)

# Look at the tuning results
# Note that ROC was the performance criterion used to select the optimal model.   


gbm.tune$bestTune
plot(gbm.tune)  		# Plot the performance of the training models
res <- gbm.tune$results
res


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
                 nrounds = 500, 
                 verbose = 1,
                 print_every_n=5,
                 early_stopping_rounds    = 15,
                 watchlist           = watchlist,
                 maximize            = FALSE,
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
                                               