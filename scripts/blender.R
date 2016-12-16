# This script blends out of bag estimates using lm to estimate weights.
# models is a data frame where the first column is y and the remaining
# columns are the oob estimates of the models to be blended

require(readr)

# Create data frame models here:
train <- read_csv('./source/train.csv.zip')
train_y <- train$loss
rm(train)
xgb1 <- readRDS('./cache/XGBbag1209A.RDS')
xgb2 <- readRDS('./cache/XGBbag1209B.RDS')
keras1oob <- read_csv('./cache/preds_oob1109B.csv')
keras2oob <- read_csv('./cache/preds_oob1205B.csv')

models <- data.frame(y = train_y, 
                     keras1 = keras1oob$loss,
                     keras2 = keras2oob$loss,
                     xgb1 = xgb1[[1]],
                     xgb2 = xgb2[[1]])

# Estimate model weights using lm with no intercept and reweight to 1.0
wts_lm <- lm(y ~ . -1, data = models)
wts <- wts_lm$coefficients / sum(wts_lm$coefficients)

# Create test data predictors
keraspred1 <- read_csv('./submissions/submission_keras1109B.csv')
keraspred2 <- read_csv('./submissions/submission_keras1205B.csv')

td <- data.frame(keras1 = keraspred1$loss,
                       keras2 = keraspred2$loss,
                       xgb1 = xgb1[[2]],
                       xgb2 = xgb2[[2]])

# Calculate weighted average prediction
test <- read_csv('./source/test.csv.zip')
testpred <- (td$keras1 + td$keras2 + td$xgb1 + td$xgb2)/4
submission <- data.frame(id = test$id, loss = testpred)
write_csv(submission, './submissions/stack_submission1209B.csv')

