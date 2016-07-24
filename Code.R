library(rpart)
library(randomForest)
#install.packages('rattle')
#install.packages('rpart.plot')
#install.packages('RColorBrewer')
library(rattle)
library(rpart.plot)
library(RColorBrewer)

setwd("~/Desktop/Data Science Dojo/")

# Load the training & testing data into R
train.org <- read.csv("Train.csv")
test.org <- read.csv("Test.csv")

# Exploring the data to see what is present
summary(train.org)



### Feature Engineering

# We see that there is no variable called child nor is there is split on the Titles. We try to create these variables. We combine the test and train data to make the transformations easier. We will later split them back into test and train data.


# Feature engineering - creating child varaible 
# #1 Combine the dataset into one large data
test.org$Survived <- NA
data <- rbind(train.org, test.org)
data$Pclass <- as.factor(data$Pclass)

# Splitting the name to get the salutation
data$Name <- as.character(data$Name)
data$Title <- sapply(data$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

# Stripping the space from the Title
data$Title <- sub(' ', '', data$Title)

# Combining the unusal names into more broad groups
data$Title[data$Title %in% c('Mme', 'Mlle', 'Ms')] <- 'Miss'
data$Title[data$Title %in% c('Capt', 'Col', 'Don', 'Major', 'Sir')] <- 'Sir'
data$Title[data$Title %in% c('Dona', 'Jonkheer', 'Lady', 'the Countess')] <- 'Lady'

data$Title <- as.factor(data$Title)

# Checking the unique values and their distribution of Titles
table(data$Title)

# Finding if the Titles have any correlation wiht the Survival variable
table(data$Survived, data$Title)


# We also realize that there are NAs in the Age variable and we try to impute those with the use of decision trees prediction.

# There are missing values in the Embarked variable as well and we replace them with the mode i.e. "S" as the place of embarkment.

# Also there is a single missing value in the Fare data we will impute that with the mean of the entire data


# Imputing the NAs in Age and Embarked with mean and median values from the entire data
Age.fit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title, data=data[!is.na(data$Age),], method="anova")
data$Age[is.na(data$Age)] <- predict(Age.fit, data[is.na(data$Age),])

# Change the Embarked variable into character and then replace it with the mode "S"
data$Embarked <- as.character(data$Embarked)
data$Embarked[data$Embarked %in% c("")] <- "S"

# Convert the variable Embarked back to factors
data$Embarked <- as.factor(data$Embarked)

# Impute the NA in fare variable
data$Fare <- ifelse(is.na(data$Fare), mean(data$Fare, na.rm = TRUE), data$Fare)


# Now that there are no NAs in the Age varaible, we can create a new feature called Child to separate the children from the adults as we know that children and females prefered in the life boats than adult males. This means that creating a child variable might help our model do better as children had a higher survival rate. We classify every person below 18 years of age as a child.

# Creating the Child variable
data$Child <- as.factor(ifelse(data$Parch >0 & data$Age < 18, 1, 0))


# We can also create a new variable that states the family size to see if larger families had trouble surviving which might very well be the case.
# Creating Familysize variable
data$FamilySize <- data$Parch + data$SibSp + 1
data$FamilySize <- as.factor(ifelse(data$FamilySize < 2, "Small", "Large"))


### Logistic Regression

# Sometimes the most simple models turn to be the best. Hence trying to use the logistic regression.

# Creating test and train data again
train <- data[1:891,]
test <- data[892:1309,]

# Fitting the logistic regression model
log.fit <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + Child + FamilySize, data = train , family = "binomial")

# Predicting using logistic model
log.pred <- predict(log.fit, test, type = "response")
log.pred <- ifelse(log.pred > 0.55, 1, 0)
log.submit <- data.frame("PassengerId" = test$PassengerId, "Survived" = log.pred)
write.csv(log.submit, "log_submit.csv", row.names = FALSE)


# As we see that this model performed exactly like that of the conditional Forest algorithm scoring the same 0.79904 on the leaderboard.


### First decision Tree model

# We start off with using a simple decision tree in the 'rpart' package and see where we stand in the Kaggle leaderboard. Before that we will split our combined data into test and train again.

# Fitting our model
rpart.fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Child + Title + Parch + Embarked + Fare + FamilySize, data = train, method = "class")

# Visualizing the rpart fit
fancyRpartPlot(rpart.fit)

# Predicting on test dataset
rpart.pred <- predict(rpart.fit, test, type = "class")
rpart.submit <- data.frame("PassengerId" = test$PassengerId, "Survived" = rpart.pred)
write.csv(rpart.submit, "rpart_submit.csv", row.names = FALSE)


# From this submission we get a score of 0.77990 which is worse than the simple logistic regression model.

### Fitting a randomForest ensemble model

# We try to fit a randomForest model and see if that outperforms our rpart % logistic model.

set.seed(1)
# Fitting the randomForest Model
rf.fit <- randomForest(as.factor(Survived) ~ Age + Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + Child + FamilySize, data = train, ntree = 2000, importance = TRUE)

# Predicting based on the RF model
rf.pred <- predict(rf.fit, test)
rf.submit <- data.frame("PassengerId" = test$PassengerId, "Survived" = rf.pred)
write.csv(rf.submit, "rf_submit.csv", row.names = FALSE)

# We find that the score is unchanged with the randomForest model and hence we will try to make use of the conditional forests

### Using Conditional Forests to make the model

# We will use the conditional forests from the package party as it uses statistical tests to make the splits.

set.seed(1)
# Installing the party package
#install.packages("party")
library(party)

# Fitting the cForest model
cF.fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize, data = train, controls=cforest_unbiased(ntree=2000, mtry=3))

# Predicting the outcomes
cF.pred <- predict(cF.fit, test, OOB=TRUE, type = "response")
cF.submit <- data.frame("PassengerId" = test$PassengerId, "Survived" = cF.pred)
write.csv(cF.submit, "cF_submit.csv", row.names = FALSE)


# This gives us a better prediction on Kaggle with a score of 0.80383 which is an improvement over our previous prediction.
