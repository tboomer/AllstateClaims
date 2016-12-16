# This script takes dataframe of the training set and predictor variable (y) in quotes, 
# and number of # bags (n) and trains a model with each bag to create a vector 
# of oob predictions.
library(Metrics)

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

set.seed(47)
train_bags <- function(df, y, n) {
    require(caret)
    require(dplyr)
    
    
    colpos <- which(colnames(df) == y)
    folds <- createFolds(unlist(df[, colpos]), k = n, list = FALSE)
    pred_train_oob <- vector("numeric", length = nrow(df))
    errors <- vector("numeric", n)
    pred_test <- matrix(data = NA, nrow = nrow(test), ncol = n)
    
    for(i in 1:n){
        inbag <- folds != i
        outbag <- folds == i
        train_X <- df[inbag, -colpos]
        train_y <- unlist(df[inbag, colpos], use.names = FALSE)
        oob_x <- df[outbag, -colpos]
        oob_y <- unlist(df[outbag, colpos], use.names = FALSE)
        
        # INSERT train commands here taking train_X as X and train_y as y
        # --------------------------------------------------------------
        features=names(train_X)
        
        trainX <- data.matrix(select(train_X, -loss, -id))
        
        ctrl <- trainControl(method = "none", 
                             summaryFunction = mae_metric,
                             verboseIter = TRUE,
                             allowParallel = TRUE)
        
        tune <- data.frame(nrounds = 800,
                           max_depth = 7,
                           eta = 0.05,
                           gamma = 1.5,
                           colsample_bytree = 0.95,
                           min_child_weight = 1,
                           subsample = 0.9)
        
        model <- train(x = trainX, y = train_y,
                       method = "xgbTree",
                       trControl = ctrl,
                       tuneGrid = tune,
                       metric = "MAE",
                       maximize = FALSE,
                       tuneLength = 1)
        # END of train commands
        #-----------------------------------------------------------------------
        #
        # Predict oob values and calculate error for current bag
        pred <- predict(model, xgb.DMatrix(data=data.matrix(oob_x[, 2:131])))
        pred <- exp(pred) - 1
        oob_y <-exp(oob_y) - 1
        mae <- sum(abs(oob_y - pred))/length(oob_y)
        cat("MAE fold ", i, mae, "\n")
        
        # Incorporate values from current bag into full oob vector
        pred_train_oob[outbag] <- pred
        errors[i] <- mae
        
        # Make test set prediction based on current bag
        pred_test[,i] <- predict(model, xgb.DMatrix(data=data.matrix(test[, 2:131])))
        pred_test[,i] <- exp(pred_test[,i]) - 1
    }
    
    # Average test predictions
    pred_test <- apply(pred_test, 1, mean)
    cat("Mean error: ", mean(errors))
    
    return(list(pred_train_oob, pred_test, errors))
}

################################################################################
source('./scripts/prep_data.R')
XGBbag1209B <- train_bags(train, "logloss", 5)
