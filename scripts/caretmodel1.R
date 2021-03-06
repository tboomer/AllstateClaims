# Run XGBoost inside caret wrapper using 5-fold CV
library(readr)
library(caret)
library(plyr)
library(dplyr)
library(xgboost)
library(Metrics)

# Define cost functions
# Custom MAE metric in caret format
mae_metric <- function (data,
                        lev = NULL,
                        model = NULL) {
    out <- mae(exp(data$obs), exp(data$pred))
    names(out) <- "MAE"
    out
}


# Calculate mean average error
calc_mae <- function(preds, y) {
    sum(abs(y - preds))/length(y)
}


source('./scripts/prep_data.R')
trainX <- data.matrix(select(train, -loss, -logloss, -id))



ctrl <- trainControl(method = "cv", 
                     number = 5,
                     summaryFunction = mae_metric,
                     verboseIter = TRUE,
                     allowParallel = TRUE)

tune <- expand.grid(nrounds = 800,
             max_depth = c(6,7,8),
             eta = 0.05,
             gamma = 1.5,
             colsample_bytree = 0.95,
             min_child_weight = 1,
             subsample = 0.9)

set.seed(111)
xgb_model4 <- train(x = trainX, y = train$logloss,
                  method = "xgbTree",
                  trControl = ctrl,
                  tuneGrid = tune,
                  metric = "MAE",
                  maximize = FALSE,
                  tuneLength = 1)
#-------------------------------------------------------------------------------
# Make prediction on train data
trainpred <- predict(xgb_model, data.matrix(train[,2:131]))
trainpred <- exp(trainpred)



# Make prediction on test data
testpred <- predict(xgb_model, data.matrix(test[,2:131]))
testpred <- exp(testpred)

submission <- data.frame(id = test$id, loss = testpred)
# write_csv(submission, './submissions/submission1207D.csv')
