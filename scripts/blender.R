# This script blends out of bag estimates using lm to estimate weights.
# models is a data frame where the first column is y and the remaining
# columns are the oob estimates of the models to be blended

# Create data frame models here:
train <- read_csv('./source/train.csv.zip')
load('./cache/xgb_oob_1.RData')
load('./cache/keras_oob_1027.RData')
models <- data.frame(y = train$loss, xgb = xgb_oob_1, nnet = g$loss)

# Estimate model weights using lm with no intercept
wts_lm <- lm(y ~ . -1, data = models)
wts <- wts_lm$coefficients / sum(wts_lm$coefficients)

# render estimate as weighted sum