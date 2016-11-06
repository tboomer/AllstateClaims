# This script takes dataframe of the training set and predictor variable (y) in quotes, 
# and number of # bags (n) and trains a model with each bag to create a vector 
# of oob predictions.

train_bags <- function(df, y, n) {
    require(caret)
    require(dplyr)
    
    colpos <- which(colnames(df) == y)
    folds <- createFolds(unlist(df[, colpos]), k = n)
    pred_oob <- vector("numeric", length = nrow(df))
    errors <- vector("numeric", n)
    
    for(i in 1:n){
        inbag <- unlist(folds[-i])
        outbag <- unlist(folds[i])
        train_X <- df[inbag, -colpos]
        train_y <- unlist(df[inbag, colpos])
        oob_x <- df[outbag, -colpos]
        oob_y <- unlist(df[outbag, colpos])
        
        # INSERT train commands here taking train_X as X and train_y as y
        # --------------------------------------------------------------
        features=names(train)
        
        
        dtrain <- xgb.DMatrix(data=data.matrix(train_X),
                              label=data.matrix(train_y))
        #watchlist<-list(dtrain = dtrain)
        
        Sys.time()
        set.seed(575)
        clf <- xgb.train(params=list(  objective="reg:linear", 
                                       booster = "gbtree",
                                       eta=0.1, 
                                       max_depth=7, 
                                       subsample=0.85,
                                       colsample_bytree=0.7) ,
                         data = dtrain, 
                         nrounds = 50, 
                         verbose = 0,
                         print_every_n=5,
                         early_stopping_rounds    = 15,
                         #watchlist           = watchlist,
                         maximize            = FALSE,
                         feval = xg_eval_mae
        )
        # END of train commands
        #-----------------------------------------------------------------------
        #
        # Predict oob values and calculate error for current bag
        pred <- predict(clf, xgb.DMatrix(data=data.matrix(oob_x)))
        pred <- exp(pred) - 1
        mae <- sum(abs(oob_y - pred))/length(oob_y)
        cat("MAE fold ", i, mae, "\n")
        
        # Incorporate values from current bag into full oob vector
        pred_oob[outbag] <- pred
        errors[i] <- mae
    }
    cat("Mean error: ", mean(errors))
    return(pred_oob)
}




