# Neural net model using train/validation set split to diagnose model performance.

library(readr)
library(caret)
library(dplyr)
library(mxnet)

# Define cost functions
# Cost function in XGBoost format 
mae_metric <- mx.metric.custom("mae", function(label, pred) {
    res <- mean(abs(exp(label) - exp(pred)))
    return(res)
})


# Calculate mean average error
calc_mae <- function(preds, y) {
    sum(abs(y - preds))/length(y)
}

source('./scripts/prep_data.R')

# Split train and val data
index <- createDataPartition(train$logloss, p = .8, list = FALSE)
val <- train[-index,]
train <- train[index,]


# Train mxnet model
trainX <- data.matrix(select(train, -loss, -logloss, -id))
trainY <- train$logloss
valX <- data.matrix(select(val, -loss, -logloss, -id))
valY <- val$logloss


data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)

lro <- mx.symbol.LinearRegressionOutput(fc1)

set.seed(57)
model <- mx.model.FeedForward.create(lro, X=trainX, y=trainY,
                                     ctx=mx.cpu(), num.round=50, array.batch.size=20,
                                     learning.rate=2e-6, momentum=0.9, eval.metric=mae_metric)


pred <- predict(model, valX)
#-------------------------------------------------------------------------------
# DIAGNOSTICS

# Plot prediction vs actual
errordata <- cbind(myval, valpred, myval$loss)
ggplot(errordata, aes(myval$loss, valpred)) + geom_point() + 
    geom_abline(slope = 1, color = "blue")

# Print variable importance.
importance <- xgb.importance(feature_names = features, model = clf)
print.data.frame(importance)

