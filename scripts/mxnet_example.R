require(mlbench)
require(mxnet)

data(Sonar, package = "mlbench")

Sonar[,61] <- as.numeric(Sonar[,61])-1
train.ind <- c(1:50, 100:150)
train.x <- data.matrix(Sonar[train.ind, 1:60])
train.y <- Sonar[train.ind, 61]
test.x <- data.matrix(Sonar[-train.ind, 1:60])
test.y <- Sonar[-train.ind, 61]

mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=10, out_node=2, out_activation="softmax",
                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9, 
                ctx = mx.cpu(), eval.metric=mx.metric.accuracy)

graph.viz(model$symbol$as.json())

preds <- predict(model, test.x)
pred.label <- max.col(t(preds)) - 1
table(pred.label, test.y)

data(BostonHousing, package="mlbench")

train.ind <- seq(1, 506, 3)
train.x <- data.matrix(BostonHousing[train.ind, -14])
train.y <- BostonHousing[train.ind, 14]
test.x <- data.matrix(BostonHousing[-train.ind, -14])
test.y <- BostonHousing[-train.ind, 14]

# Define the input data
data <- mx.symbol.Variable("data")
# A fully connected hidden layer
# data: input source
# num_hidden: number of neurons in this hidden layer
fc1 <- mx.symbol.FullyConnected(data, num_hidden=1)

# Use linear regression for the output layer
lro <- mx.symbol.LinearRegressionOutput(fc1)

mx.set.seed(0)
model <- mx.model.FeedForward.create(lro, X=train.x, y=train.y,
                                     ctx=mx.cpu(), num.round=50, array.batch.size=20,
                                     learning.rate=2e-6, momentum=0.9, eval.metric=mx.metric.rmse)

preds <- predict(model, test.x)
sqrt(mean((preds-test.y)^2))