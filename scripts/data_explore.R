# Exploratory data analysis of Allstate Claims Data

library(readr)
library(ggplot2)
library(dplyr)
library(reshape2)

train <- read_csv('./source/train.csv.zip')
ggplot(train, aes(loss)) + geom_histogram(bins = 100)

plotdata <- filter(train, loss>10000)
ggplot(plotdata, aes(loss)) + geom_histogram(bins = 100)

train$logloss <- log(train$loss + 1)
ggplot(train, aes(logloss)) + geom_histogram(bins = 100)

sapply(train[,2:117], unique)

sapply(train, function(x) sum(is.na(x)))

ggplot(train, aes(cat110)) + geom_bar()
ggplot(train, aes(cont1)) + geom_histogram()
ggplot(train, aes(cont2)) + geom_histogram()
ggplot(train, aes(cont3)) + geom_histogram()
ggplot(train, aes(cont4)) + geom_histogram()
ggplot(train, aes(cont5)) + geom_histogram()
ggplot(train, aes(cont6)) + geom_histogram()
ggplot(train, aes(cont7)) + geom_histogram()
ggplot(train, aes(cont8)) + geom_histogram()
ggplot(train, aes(cont9)) + geom_histogram()
ggplot(train, aes(cont10)) + geom_histogram()
ggplot(train, aes(cont11)) + geom_histogram()
ggplot(train, aes(cont12)) + geom_histogram()
ggplot(train, aes(cont13)) + geom_histogram()
ggplot(train, aes(cont14)) + geom_histogram()


cormat <- round(cor(train[,118:131]),2)
melt_cormat <- melt(cormat)
ggplot(data = melt_cormat, aes(x=Var1, y=Var2, fill=value)) +
geom_tile()


ggplot(mytrain, aes(x=id, y=logloss)) + geom_point(size = 0.1)
# No correlation with ID
#-------------------------------------------------------------------------------

