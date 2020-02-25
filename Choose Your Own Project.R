if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

#Loading the dataset 
data<- read_csv("https://raw.githubusercontent.com/nadimyatim/Choose-Your-Own-Project/master/adult.csv")


#The number of observations and features in the data
dim(data)

#The classes of each feature
class(data$age)
class(data$workclass)
class(data$fnlwgt)
class(data$education)
class(data$education.num)
class(data$marital.status)
class(data$occupation)
class(data$relationship)
class(data$race)
class(data$sex)
class(data$capital.gain)
class(data$capital.loss)
class(data$hours.per.week)
class(data$native.country)
class(data$income)

#The composition of some of the features having the class "character"
unique(data$workclass)
unique(data$education)
unique(data$education.num)
unique(data$marital.status)
unique(data$occupation)
unique(data$relationship)
unique(data$sex)
unique(data$race)
unique(data$income)


#showing a sample of the  dataset
head(data) 

#showing summary statistics of the  dataset and its features.
summary(data) 

#Impact of Hours per week on income
data %>% ggplot(aes(income,hours.per.week)) + geom_boxplot(fill="navy",col="red") 

#Impact of age on income
  data %>% ggplot(aes(income,age)) + geom_boxplot(fill="navy",col="red") 
  
#Impact of education on income
  data %>% ggplot(aes(income,education.num)) + geom_boxplot(fill="navy", col="red") 
  
#Impact of capital gain on income
  data %>% ggplot(aes(income,capital.gain)) + geom_boxplot(fill="navy",col="red") 
  
#Impact of capital loss on income
  data %>% ggplot(aes(income,capital.loss)) + geom_boxplot(fill="navy",col="red") 
  
#Impact of fnlwgt based on income
  data %>% ggplot(aes(income,fnlwgt)) + geom_boxplot(fill="navy",col="red")   

#Distribution of workclass based on income
  data %>% ggplot(aes(workclass,fill=income)) + geom_bar(col="navy") + theme(axis.text.x = element_text(angle = 60, hjust = 1))

#Distribution of race based on income
  data %>% ggplot(aes(race,fill=income)) + geom_bar(col="navy") + theme(axis.text.x = element_text(angle = 60, hjust = 1))

#Distribution of sex based on income
  data %>% ggplot(aes(sex,fill=income)) + geom_bar(col="navy") + theme(axis.text.x = element_text(angle = 60, hjust = 1))
  
#Distribution of marital status based on income
  data %>% ggplot(aes(marital.status,fill=income)) + geom_bar(col="navy") + theme(axis.text.x = element_text(angle = 60, hjust = 1))
  
#Distribution of relationship based on income
  data %>% ggplot(aes(relationship,fill=income)) + geom_bar(col="navy") + theme(axis.text.x = element_text(angle = 60, hjust = 1))
  
#Distribution of occupation based on income
  data %>% ggplot(aes(occupation,fill=income)) + geom_bar(col="navy") + theme(axis.text.x = element_text(angle = 60, hjust = 1))
  
  
#Data Splitting into Training and Test Sets
  
 #Checking if there are any NAs in our dataset
 if_NA<- data %>% anyNA(data)
 #Removing Nas from the dataset
 data <- na.omit(data)
  
 #Setting the seed to 1
  set.seed(1)
 #Creating data partitions and splitting the dataset
  test_index <- createDataPartition(data$income, times = 1, p = 0.1, list = FALSE)
  train_set <- data[-test_index,]
  test_set <- data[test_index,]
  
  
#Logistic regression Model
  #Training the model
  train_glm <- train(income ~ ., 
                         method = "glm", 
                         data = train_set)
  #Caclulating the accuracy and constructing the confusion matrix
  glm_preds <- predict(train_glm, test_set)
  glm_accuracy<- mean(glm_preds == test_set$income)
  confusionMatrix(factor(glm_preds), reference = factor(test_set$income))
  #Calculating the F1 score
  glm_F1<- F_meas(factor(glm_preds), factor(test_set$income))
  #Creating the data frame Model_Results
  Model_Results<- data_frame(Model="Logistic Regression",
                             Accuracy=glm_accuracy, 
                             F1score= glm_F1)
  
  #Displaying the model results
  Model_Results %>% knitr::kable()
  

  
  
#Linear Discriminate Analysis model
  #Training the model
  train_lda <- train(income ~ ., 
                     method = "lda", 
                     data = train_set)  
  #Caclulating the accuracy and constructing the confusion matrix
  lda_preds <- predict(train_lda, test_set)
  lda_accuracy<- mean(lda_preds == test_set$income)
  confusionMatrix(factor(lda_preds), reference = factor(test_set$income))
  #Calculating the F1 score
  lda_F1<- F_meas(factor(lda_preds), factor(test_set$income))
  #Updating the data frame Model_Results
  Model_Results<- bind_rows(Model_Results,
                            data_frame(Model="Linear Discriminant Analysis",
                                       Accuracy=lda_accuracy, 
                                       F1score= lda_F1))
  
  
  #Displaying the model results
  Model_Results %>% knitr::kable()
  
  
#Decision Tree Model
  #Training the model
  train_rpart <- train(income ~ ., 
                       method = "rpart", 
                       data = train_set)
  #Caclulating the accuracy and constructing the confusion matrix
  rpart_preds <- predict(train_rpart, test_set)
  rpart_accuracy<- mean(rpart_preds == test_set$income)
  confusionMatrix(factor(rpart_preds), reference = factor(test_set$income))
  #Calculating the F1 score
  rpart_F1<- F_meas(factor(rpart_preds), factor(test_set$income))
  #Updating the data frame Model_Results
  Model_Results<- bind_rows(Model_Results,
                            data_frame(Model="Decision Tree",
                                       Accuracy=rpart_accuracy, 
                                       F1score= rpart_F1))
  
  
  #Displaying the model results
  Model_Results %>% knitr::kable()


#Random Forest Model
  #Training the model
  train_rforest <- train(income ~ ., 
                       method = "rf",
                       data = train_set, 
                       ntree= 7,
                       importance=TRUE)
  #Caclulating the accuracy and constructing the confusion matrix
  rforest_preds <- predict(train_rforest, test_set)
  rforest_accuracy<- mean(rforest_preds == test_set$income)
  confusionMatrix(factor(rforest_preds), reference = factor(test_set$income))
  #Calculating the F1 score
  rforest_F1<- F_meas(factor(rforest_preds), factor(test_set$income))
  #Updating the data frame Model_Results
  Model_Results<- bind_rows(Model_Results,
                            data_frame(Model="Random Forest",
                                       Accuracy=rforest_accuracy, 
                                       F1score= rforest_F1))
  

  #Displaying the model results
  Model_Results %>% knitr::kable()
 
#Ensemble Model
  #Caclulating the accuracy and constructing the confusion matrix
  ensemble <- cbind(glm = glm_preds=="<=50K" , lda = lda_preds=="<=50K", decision=rpart_preds=="<=50K", randomforest=rforest_preds=="<=50K")
  
  ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "<=50K", ">50K")
  ensemble_accuracy<-mean(ensemble_preds == test_set$income)
  confusionMatrix(factor(ensemble_preds), reference = factor(test_set$income))
  #Calculating the F1 score 
  ensemble_F1<- F_meas(factor(ensemble_preds), factor(test_set$income))
  
  #Updating the data frame Model_Results
  Model_Results<- bind_rows(Model_Results,
                            data_frame(Model="Ensemble",
                                       Accuracy=ensemble_accuracy, 
                                       F1score= ensemble_F1))
  
  #Displaying the model results
  Model_Results %>% knitr::kable()