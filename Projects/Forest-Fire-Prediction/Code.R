library(readxl)
ForestFiresWith <- read_excel("C:/Users/16023/Desktop/IST 707/ForestFiresWith.xlsx")
library(corrplot)
library(corrgram)

ff <- ForestFiresWith
View(ff)
corrplot(ff, method = "number")
corrplot(corrgram(ff))

summary(ff)
ff <- ff[,-13]
str(ff)
sapply(ff, sd)
trainRatio <- .67
set.seed(1016) # Set Seed so that same sample can be reproduced in future also
sample <- sample.int(n = nrow(ff), size = floor(trainRatio*nrow(ff)), replace = FALSE)
ff$X <- log(ff$X)
testdata <- ff[-sample, ]
testdata
testdata <- testdata[, -c(3:4)]
summary(testdata)
traindata <- ff[sample, ]
traindata <- traindata[, -c(3:4)]
summary(traindata)
probit2 <- glm(traindata$fire__no_yes ~.,  family = binomial(link = "probit"), data=traindata[,-(length(traindata))])
logit2 <- glm(traindata$fire__no_yes ~.,  family = "binomial", data=traindata[,-(length(traindata))])
summary(probit2)
summary(logit2)

predictedlogit <- plogis(predict(logit2, testdata))
predictedprobit <- plogis(predict(probit2, testdata))
table(predictedlogit > 0.5, testdata$fire__no_yes)

library(e1071)
library(caret)
### SVM Model
svmclassifier = svm(formula = traindata2$`Y/N` ~ ., 
                    data = traindata2, 
                    type = 'C-classification', 
                    kernel = 'linear') 

y_pred <- predict(svmclassifier, newdata = testdata2[-9]) 
cm <- table(testdata2$`Y/N`, y_pred) 
cm

prediction <- predict(svmclassifier, newdata = testdata2[-9]) 
results <- data.frame(testdata2$`Y/N`, prediction) 
colnames(results) <- c("Actual", "Prediction")   
str(results) 
results$Prediction <- as.factor(results$Prediction) 
results$Actual <- as.factor(results$Actual)
confusionMatrix(results$Prediction, results$Actual) 



svmclassifier2 = svm(formula = traindata2$`Y/N` ~ ., 
                     data = traindata2, 
                     type = 'C-classification', 
                     kernel = 'polynomial') 

y_pred <- predict(svmclassifier2, newdata = testdata2[-9]) 
cm <- table(testdata2$`Y/N`, y_pred) 
cm





prediction <- predict(svmclassifier2, newdata = testdata2[-9]) 
results <- data.frame(testdata2$`Y/N`, prediction) 
colnames(results) <- c("Actual", "Prediction")   
str(results) 
results$Prediction <- as.factor(results$Prediction) 
results$Actual <- as.factor(results$Actual)
confusionMatrix(results$Prediction, results$Actual) 

svmclassifier3 = svm(formula = traindata2$`Y/N` ~ ., 
                     data = traindata2, 
                     type = 'C-classification', 
                     kernel = 'sigmoid') 

y_pred <- predict(svmclassifier3, newdata = testdata2[-9]) 
cm <- table(testdata2$`Y/N`, y_pred) 
cm

prediction <- predict(svmclassifier3, newdata = testdata2[-9]) 
results <- data.frame(testdata2$`Y/N`, prediction) 
colnames(results) <- c("Actual", "Prediction")   
str(results) 
results$Prediction <- as.factor(results$Prediction) 
results$Actual <- as.factor(results$Actual)
confusionMatrix(results$Prediction, results$Actual) 

svmclassifier4 = svm(formula = traindata2$`Y/N` ~ ., 
                     data = traindata2, 
                     type = 'C-classification', 
                     kernel = 'radial') 

y_pred <- predict(svmclassifier4, newdata = testdata2[-9]) 
cm <- table(testdata2$`Y/N`, y_pred) 
cm

prediction <- predict(svmclassifier4, newdata = testdata2[-9]) 
results <- data.frame(testdata2$`Y/N`, prediction) 
colnames(results) <- c("Actual", "Prediction")   
str(results) 
results$Prediction <- as.factor(results$Prediction) 
results$Actual <- as.factor(results$Actual)
confusionMatrix(results$Prediction, results$Actual) 

library(ggplot2)
library(dplyr)
library(tidyr)
library(rpart)
library(rpart.plot)
###created XY coordinates and adding it to ff
df <- paste(ff$X,",",ff$Y)
df <- as.data.frame(df)
colnames(df) <- "coordinates"
df
ff2 <- cbind(ForestFiresWith ,df)
ff2 <- as.data.frame(ff2) 
## no need to X and Y columns when we have x,y column
ff2 <- ff2[,c(5:15)]
str(ff2)
View(ff2)
## Using tapply to sum up the total area burn per coordinate X,Y
areaburnedbycoord <- tapply(ff2$area, ff2$coordinates, FUN = sum)
areaburnedbycoord <- cbind(coordinates = rownames(areaburnedbycoord), areaburnedbycoord)
colnames(areaburnedbycoord) <- c("coordinates", "total_area_burned")
areaburnedbycoord <- as.data.frame(areaburnedbycoord)
summary(areaburnedbycoord)
barburn <- ggplot(data = areaburnedbycoord, aes(x= areaburnedbycoord$coordinates, y = areaburnedbycoord$total_area_burned))
barburn <- barburn + geom_bar(stat = "identity", width = .5, color = "black", size =1)
barburn <- barfreq + ggtitle("Total Area Burnt")  + labs(y="Hectares Burnt", x = "Coordinate plane")
barburn
##This code did not work
## Using tapply to sum up the total times there was a fire per coordinate X,Y
##ff2 <- ff2 %>% filter(ff2$fire__no_yes != 0)
##str(ff2)
##freqofburnedarea <- tapply(as.numeric(ff2$fire__no_yes), ff2$coordinates, FUN = length)
##freqofburnedarea <- cbind(coordinates = rownames(freqofburnedarea), freqofburnedarea)
##colnames(freqofburnedarea) <- cbind("coordinates", "freq_of_fires")


freqofburnedarea <- ffFreq[1:36,1:2]
colnames(freqofburnedarea) <- cbind("coordinates", "freq_of_fires")
summary(freqofburnedarea)
sum(freqofburnedarea$freq_of_fires)
sum(ff2$fire__no_yes)
barfreq <- ggplot(data = freqofburnedarea, aes(x= freqofburnedarea$coordinates, y = freqofburnedarea$freq_of_fires)) 
barfreq <- barfreq + geom_bar(stat = "identity", width = .5, color = "black", size =1)
barfreq <- barfreq + ggtitle("Number of Fires")  + labs(y="Frequency", x = "Coordinate plane")
barfreq
# Decistion Tree
ff3 <- ff2[,-c(9,11)]
View(ff3)
ff3
set.seed(123)
index1 <- sample(1:nrow(ff3), round(nrow(ff3)*.8))
ff4 <- ff3[index1,]
dt <- rpart(ff4$fire__no_yes ~., method = 'class', data = ff4)
dt
printcp(dt) # display the results
plotcp(dt) # visualize cross-validation results
summary(dt)
rpart.plot(dt, box.palette="RdBu", shadow.col="gray", nn=TRUE)

y_pred = predict(dt, newdata = ff4[,-9])
rfp <- as.data.frame(y_pred)
rfp$y_pred <- as.factor(rfp$y_pred)
rfa <- as.data.frame(ff4[,9])
rfa$`Y/N` <- as.factor(rfa$`Y/N`)
length(rfa$`Y/N`)
length(rfp$y_pred)
confusionMatrix(rfp$y_pred, rfa$`Y/N`)


set.seed(1016)
index2 <- sample(1:nrow(ff3), round(nrow(ff3)*.8))
ff5 <- ff3[index2,]
dt <- rpart(ff5$fire__no_yes ~., method = 'class', data = ff5)
dt
printcp(dt) # display the results
plotcp(dt) # visualize cross-validation results
summary(dt)
rpart.plot(dt, box.palette="RdBu", shadow.col="gray", nn=TRUE)

set.seed(69)
index3 <- sample(1:nrow(ff3), round(nrow(ff3)*.8))
ff6 <- ff3[index3,]
dt <- rpart(ff6$fire__no_yes ~., method = 'class', data = ff6)
dt
printcp(dt) # display the results
plotcp(dt) # visualize cross-validation results
summary(dt)
rpart.plot(dt, box.palette="RdBu", shadow.col="gray", nn=TRUE)



#### Random Forest
install.packages("randomForest")
library(randomForest)
traindata2 <- traindata
testdata2 <- testdata
colnames(traindata2) <- c("FFMC","DMC","DC", "ISI","Temp","Rel_Hum","Wind_Speed","Rain_Amt","Y/N")
colnames(testdata2) <- c("FFMC","DMC","DC", "ISI","Temp","Rel_Hum","Wind_Speed","Rain_Amt","Y/N")
traindata2$`Y/N` <- as.factor(traindata2$`Y/N`)
traindata2 <- as.data.frame(traindata2)
Random_Forest <- randomForest(formula = traindata2$`Y/N` ~ .,data=traindata2,ntree = 400, mtry = 6, importance = TRUE)
Random_Forest
plot(Random_Forest)
gt <- getTree(Random_Forest, 5, labelVar=TRUE)
gt <- as.data.frame(gt)
summary(gt)

y_pred = predict(Random_Forest, newdata = testdata2[,-9])
rfp <- as.data.frame(y_pred)
rfp$y_pred <- as.factor(rfp$y_pred)
rfa <- as.data.frame(testdata2[,9])
rfa$`Y/N` <- as.factor(rfa$`Y/N`)
length(rfa$`Y/N`)
length(rfp$y_pred)
confusionMatrix(rfp$y_pred, rfa$`Y/N`)


hist(ff3$fire__no_yes)

ggplot(ff3, aes(x=ff3$fire__no_yes)) + geom_histogram(binwidth=0.4) + labs(title = "Severe Fire Yes or No", x="No                                                                          Yes", y="count")




library(corrplot)
library(corrgram)
library(readr)
library(e1071)
library(caret)
library(ggplot2)
library(dplyr)
install.packages("taRifx")
library( taRifx )
install.packages("GGally")
library(GGally)
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





#testdata$fire__no_yes <- as.factor(testdata$fire__no_yes)

x<-traindata[ , -which(names(traindata) %in% c("fire__no_yes"))]
str(x)
y <- traindata[,"fire__no_yes"]
str(y)

#y <-sapply(y,as.factor)
model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
train_naibayes <- naiveBayes(traindata2$`Y/N` ~., data=traindata2, na.action = na.pass)
train_naibayes
str(model)

#Model Evaluation
#Predict testing set
Predict <- predict(model,newdata = testdata )
#Get the confusion matrix to see accuracy value and other parameter values
Predict
confusionMatrix(Predict, traindata$fire__no_yes )

library(pROC)
library(CORElearn)
library(RWeka)
library(FSelector)
str(ff6)
ff6$fire__no_yes <-  as.factor(ff6$fire__no_yes)
IG.CORElearn <- attrEval(ff6$fire__no_yes ~ ., data=ff6,  estimator = "InfGain")
IG.RWeka     <- InfoGainAttributeEval(Species ~ ., data=iris,)
IG.FSelector <- information.gain(Species ~ ., data=iris,)
IG.CORElearn
