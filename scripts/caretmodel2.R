# Run XGBoost inside caret wrapper single model
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



ctrl <- trainControl(method = "none",
                     summaryFunction = mae_metric,
                     verboseIter = TRUE,
                     allowParallel = TRUE)

tune <- data.frame(nrounds = 800,
             max_depth = 7,
             eta = 0.05,
             gamma = 1,
             colsample_bytree = 0.9,
             min_child_weight = 1,
             subsample = 1)

set.seed(555)
xgb_model <- train(x = trainX, y = train$logloss,
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
write_csv(submission, './submissions/submission1207B.csv')
