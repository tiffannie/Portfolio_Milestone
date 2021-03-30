library(corrplot)
library(corrgram)
library(readr)
library(e1071)
library(caret)
library(ggplot2)
library(dplyr)
library( taRifx )
library(GGally)
library(pROC)

#ff <- ForestFiresWith
#ff <-as.data.frame(read.csv("C:/Users/tmacd/Downloads/fire.csv"))
ff<-read.csv("C:/Users/tmacd/Downloads/fire.csv")
#ff <-as.data.frame(ff)

#fff<-ff %>% mutate_if(is.numeric,funs(as.factor)) 
str(ff)

#corrplot(ff, method = "number")
corrplot(corrgram(ff))

str(ff)
ff$month <- as.factor(ff$month)
ff$day <- as.factor(ff$day)
ff$fire__no_yes <- as.factor(ff$fire__no_yes)

ff <- japply( ff, which(sapply(ff, class)=="integer"), as.numeric )
str(ff)

numff<-ff[,-c(1,2,3,4)]
str(numff)
ggpairs(numff)

#remove "area" column.
ff <- ff[,-13]
#str(ff)
#sapply(ff, sd)
trainRatio <- .67
set.seed(1016) # Set Seed so that same sample can be reproduced in future also
sample <- sample.int(n = nrow(ff), size = floor(trainRatio*nrow(ff)), replace = FALSE)
testdata <- ff[-sample, ]
str(testdata)
testdata <- testdata[, -c(1:4)]
summary(testdata)
traindata <- ff[sample, ]
traindata <- traindata[, -c(1:4)]
summary(traindata)

#View(traindata)

traindata2 <- traindata
colnames(traindata2) <- c("FFMC","DMC","DC", "ISI","Temp","Rel_Hum","Wind_Speed","Rain_Amt","Y/N")
traindata2$`Y/N` <- as.factor(traindata2$`Y/N`)
traindata2 <- as.data.frame(traindata2)


train_naibayes <- naiveBayes(traindata2$`Y/N` ~., data=traindata2, na.action = na.pass)

str(traindata2)

#removing yes/no label to test
testdata2 <- testdata[,-9]

#Naive Bayes model Prediction
nb_Pred <- predict(train_naibayes,testdata2)
nb_Pred

testsdata2 <- testdata[,-9]

#Testing accurancy of naive bayes model with Kaggle train data sub set
(confusionMatrix(nb_Pred, testdata$fire__no_yes))



#Plot Variable performance
X <- varImp(train_naibayes)
X
plot(X)
