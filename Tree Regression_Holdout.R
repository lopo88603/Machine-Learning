# SETTING DIRECTORY

rm(list = ls())
setwd("C:\\Users\\lopo\\OneDrive\\桌面\\course\\statistical machine learning\\project\\")

#INSTALLING PACKAGES

install.packages("data.table")
install.packages("plot3D")
install.packages("viridisLite")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("knitr")
install.packages("caret")

#LOADING LIBRARIES

library(data.table)
library(ggplot2)
library(plot3D)
library(viridisLite)
library(rpart)
library(rpart.plot)
library(dplyr)
library(caret) #For the repeated cross validation

#INVOKING THE DATA SET

dt_supermarkets<- fread(file="dataset.csv", sep=";")

  
#GOAL____________________________________________________________________________________________________
#We want to predict is the Rating that give people to supermarkets in Berlin (The value range between 1-5)
  #If is zero then there is no review since the minimium to review is 1
  #The Google?s Rating is the weighted average of the reviews by the people
  #There is only ratings as integers (1,2,3,4,5) is not posible to rate in decimals

  #As Independant variables we have: 
  # UAB: How many hours in all the week (business time) is full on demand (in dataset there is UAB for every day of the week)
  # UABWEEK: How many hours from Monday to Friday (business time) is full on demand
  # REVIEWS: Number of people giving a rating
  # AVGTIME: Average time spend on the supermarket by people
  # DISTCENT: Distance in km to the center
  # Open_Hour_i: How many hours does the supermarket is open in the day i (From Monday to Sunday)
  # NUM_COMMENTS: Number of comments in the google maps site of the supermarket
  # NEG_COMMENTPRICE: Number of negative comments related to the price in the supermarket
  # NEG_COMMENTFOOD: Number of negative comments related to the food in the supermarket
  # NEGPOS_RATIO: Ratio of how many negative comments are with respect possitive comments
  # POSNEG_RATIO: Ratio of how many positive comments are with respect negative comments
  # NEG_REVIEWS: Number of Reviews with 2 and 1 stars
  # POS_REVIEWS: Number of Reviews with 4 and 5 stars

# As dummy variables: Parking Lot in the Supermarket, Is it work 24hrs and is it working on sundays
# As a categorical variable: The opening time being 1 at 6:00am, 2 at 7:00 am, 3 at 8:00 am and 4 at 9:00 am or more

#OBSERVING THE FREQUENCY OF RATINGS
  hist(dt_supermarkets$RATING, main = "Rating Histogram", xlab = "Rating", freq=TRUE)
  avg_rating <- mean(dt_supermarkets$RATING)
  abline(v = avg_rating, col = "red", lwd = 2)
  rating_sd <- sd(dt_supermarkets$RATING)
  print(rating_sd)
    #We can observe that there is less data in less than 3.5 to 2.5 there is not less than 2
    #Consumers rate more positevily the ranking
    #The average rating is at 4.00 with a standar deviation of 0.28 is close to the mean

  
#CHECKING THE FREQUENCY OR REVIEWS BY RATING
  
  hist(dt_supermarkets$REVIEWS, main = "Reviews Histogram", xlab = "Total of reviews", freq=TRUE)
  avg_review <- mean(dt_supermarkets$REVIEWS)
  abline(v = avg_review, col = "red", lwd = 2)
  reviews_sd <- sd(dt_supermarkets$REVIEWS)
  print(reviews_sd)
  
  OLSRATING_REVIEWS <- lm(formula=RATING~REVIEWS, data=dt_supermarkets)
  summary(OLSRATING_REVIEWS)
  
  ggplot(data = dt_supermarkets, aes(y=RATING, x=REVIEWS)) +
    geom_function(fun = function(x) 3.960639e+00+5.401796e-05*x) +
    geom_point() +
    labs(
      x = "NUMBER OF REVIEWS",
      y = "RATING") +
    theme_minimal() +
    theme(axis.line = element_line(color = "#000000"))
  
  #OMITTING THE TOP 3 SUPERMARKETS WITH MORE THAN 4400 REVIEWS TO CHECK THE COEFFICIENT RELATIONSHIP
  ommittop3_dataset <- dt_supermarkets[dt_supermarkets$REVIEWS <= 4400, ]
  lm(formula=RATING~REVIEWS, data=ommittop3_dataset)
  summary(ommittop3_dataset)
  
  ggplot(data = ommittop3_dataset, aes(y=RATING, x=REVIEWS)) +
    geom_function(fun = function(x) 3.9278455+0.0001015*x) +
    geom_point() +
    labs(
      x = "NUMBER OF REVIEWS WITHOUT TOP3",
      y = "RATING") +
    theme_minimal() +
    theme(axis.line = element_line(color = "#000000"))
  
  
#OBSERVING THE CORRELATION BETWEEN VARIABLES

  OLS_supermarkets<-lm(formula=RATING~AVGTIME+DISTCENT+NUM_COMMENTS+NEG_COMMENTPRICE
                       +POS_REVIEWS +NEG_REVIEWS+perc_BH+COMCENTER+PARKLOT+OPEN6AM+OPEN7AM+OPEN8AM+OPENAFTER8, data=dt_supermarkets)
  summary(OLS_supermarkets)  
  
  #Checking the MSE (Mean Squared Error)
  
  predicted_OLS <- predict(OLS_supermarkets)
  residuals_OLS <- dt_supermarkets$RATING - predicted_OLS
  squared_residuals_OLS <- residuals_OLS^2
  mse_OLS <- mean(squared_residuals_OLS)
  
#OBSERVING HOW MUCH IMPACT HAS THE ALREADY RATINGS IN THE SUPERMARKETS
  
  ggplot(data = dt_supermarkets, aes(y=RATING, x=NEGPOS_RATIO)) +
    geom_function(fun = function(x) 4.18-1.15*x) +
    geom_point() +
    labs(
      x = "NegativePositive Ratio",
      y = "Google?s Rating") +
    theme_minimal() +
    theme(axis.line = element_line(color = "#000000"))


#ESTABLISH A TREE REGRESION WITH ANOVA (ANALISIS OF VARIANCE)_________________________________________ 
  
  #HOLD OUT division of data
  
  dt_sup<- fread(file="dataset.csv", sep=";")
  set.seed(101)
  #SPLITTING THE DATA IN TRAIN SER AND TEST SET
  training_sample <- dt_sup[, sample(.N, floor(.N*0.8))]
  dt_training <- dt_sup[training_sample]
  dt_test <- dt_sup[-training_sample]
  
  #TREE REGRESSION
  
  model_tree_sup <- rpart(data=dt_training, formula = RATING~AVGTIME+DISTCENT
                          +NUM_COMMENTS+NEG_COMMENTPRICE
                          +POS_REVIEWS +NEG_REVIEWS+perc_BH+COMCENTER+PARKLOT+
                            OPEN6AM+OPEN7AM+
                            OPEN8AM+OPENAFTER8 , method="anova")
  prp(model_tree_sup, digits = -3)
  
  summary(model_tree_sup)  
  plotcp(model_tree_sup)
  model_tree_sup$cptable[,"xerror"]
  
  #CHOOSING THE CP VALUE THAT MINIMIZES THE XERRROR AND PRUNE THE MODEL OBJECT
  
   min_cp <- model_tree_sup$cptable[which.min(model_tree_sup$cptable[, "xerror"]), "CP"]
   model_tree_supcpmin <- prune(model_tree_sup, cp = min_cp ) #the min cp is 0.01889
   prp(model_tree_supcpmin,digits = -3)
  
   model_tree_supcpmin
  
   
   
   #CALCULTING THE MSE OF THE PREDICTION WITH TRAINING DATA
   
   predTREE_Training <- predict(model_tree_supcpmin, dt_training[, c("OPENAFTER8","OPEN8AM","OPEN7AM",
                                                               "OPEN6AM","PARKLOT","NEG_REVIEWS",
                                                               "NEG_COMMENTPRICE","POS_REVIEWS",
                                                               "NUM_COMMENTS","DISTCENT","AVGTIME",
                                                               "perc_BH","COMCENTER" )])
  

    MSE_TREE_Training <- mean((dt_training$RATING - predTREE_Training)^2)
    MSE_TREE_Training
    #THE MSE_TREE TRAINING IS 0.0311967
    
    
    #CALCULTING THE MSE OF THE PREDICTION TESTING DATA
    
    predTREE_Testing <- predict(model_tree_supcpmin, dt_test[, c("OPENAFTER8","OPEN8AM","OPEN7AM",
                                                                      "OPEN6AM","PARKLOT","NEG_REVIEWS",
                                                                      "NEG_COMMENTPRICE","POS_REVIEWS",
                                                                      "NUM_COMMENTS","DISTCENT","AVGTIME",
                                                                      "perc_BH","COMCENTER" )])
    
    
    MSE_TREE_Testing <- mean((dt_test$RATING - predTREE_Testing)^2)
    MSE_TREE_Testing
    #THE MSE_TREE TESTING IS 0.0604968
    
    
    
    
    
    
    
    
    
      
    
    


  
