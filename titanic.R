library(tidyverse)
library(DataExplorer)
library(corrplot)  # for the correlation matrix
library(bestglm)  # for variable selection
library(car)  # for the VIFs
library(pROC)  # ROC Curve
library(ROCR)  # ROC Curve Color-Coded
library(lattice)
library(caret)
library(naniar)
library(textcat)
library(randomForest)
library(mice) # imputation

sz <- 22 

#Read in data
test <- read_csv("test.csv")
train <- read_csv("train.csv")

#Exploratory analysis
full  <- bind_rows(train, test) # bind training & test data
#full %>% select(Survived, Ticket, SibSp, Sex, Name, Pclass, Embarked, Cabin) %>% as.factor()
glimpse(full)
plot_missing(full)

#Missing Fare
full[which(is.na(full$Fare)),]

ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)), colour='red', linetype='dashed', lwd=1)

full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)

#Missing Embarked
full[which(is.na(full$Embarked)),]

embark_fare <- full %>%
  filter(PassengerId != 62 & PassengerId != 830)

ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), colour='red', linetype='dashed', lwd=2)

full$Embarked[c(62, 830)] <- "C"

#Missing Age
hist(full$Age)
factor_vars <- c('Survived', 'PassengerId', 'Pclass', 'Name', 'Sex', 'Embarked', 'Cabin', 'Ticket')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))
glimpse(full)
age.lm <- lm(Age ~. -Survived -Cabin -PassengerId,data = full)
missing_Age <- full[which(is.na(full$Age)),]

# Set a random seed
set.seed(129)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Survived')], method='rf') 
mice_output <- complete(mice_mod)

par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

full$Age <- mice_output$Age
plot_missing(full)

train <- full[1:891,]
test <- full[892:1309,]


#Logistic stuff
corrplot(cor(train[,c(6,7,8,10)]), type = "upper")

titanic.logistic <- glm(Survived ~., data = train[,!colnames(train) %in% c("PassengerId", "Cabin", "Name", "Ticket")], 
                      family = binomial(link = "logit"),  maxit = 100) 
log.predictions=predict(titanic.logistic, newdata=test, type="response")
log.prediction.rd <- ifelse(log.predictions > 0.5, 1, 0)

log.preds <- data.frame(PassengerId=test$PassengerId, Survived=log.prediction.rd )

write_csv(x=log.preds, path="./log.csv")



#Machine learning
gbm <- train(form=as.factor(Survived)~.,
             data=train %>% select(-Cabin),
             method="gbm",
             trControl=trainControl(method="repeatedcv",
                                    number=3, #Number of pieces of your data
                                    repeats=1) #repeats=1 = "cv"
)
length(test$PassengerId)
nrow(test)

gbm$results
gbm.preds <- data.frame(PassengerId=test$PassengerId, Survived=predict(gbm, newdata=test))

write_csv(x=gbm.preds, path="./gbm.csv")


#use xgb
xgb <- train(form=as.factor(Survived)~.,
             data=train %>% select(-Cabin),
             method="xgbTree",
             trControl=trainControl(method="repeatedcv",
                                    number=3, #Number of pieces of your data
                                    repeats=1) #repeats=1 = "cv"
)

xgb$results
xgb.preds <- data.frame(PassengerId=test$PassengerId, Survived=predict(xgb, newdata=test))

write_csv(x=xgb.preds, path="./xgbtree.csv")

