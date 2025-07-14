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
install.packages("randomForest")
install.packages("forestplot")
install.packages("gbm")

#LOADING LIBRARIES

library(data.table)
library(ggplot2)
library(plot3D)
library(viridisLite)
library(rpart)
library(rpart.plot)
library(dplyr)
library(caret) 
library(randomForest)
library(forestplot)
library(gbm)

#DEFINING TRAINING SET (80%) AND TEST DATA(20%) OF THE SAMPLE
set.seed(555)

  dt_supermarketsLTE<- fread(file="datasetLTE.csv", sep=";")
  #Create training/test set
  training_sample <- dt_supermarketsLTE[,sample(.N, floor(.N*.80))]
  dt_training <- dt_supermarketsLTE[training_sample]
  dt_test <- dt_supermarketsLTE[-training_sample]
  #the regression uses 22 variables and is different from the tree regression previously
  #the single optimal tree uses 14 variables previously revised for correlation and uses all the data
 
  #Modeling a single tree no prunning as baseline to predict the rating
  model_tree <- rpart(RATING ~ ., data=dt_training, method="anova")
  prp(model_tree, digits = -3) # reduce the number of decimal places

  predTREE_Training <- predict(model_tree, dt_training)
  
  MSE_TREE22_Training <- mean((dt_training$RATING - predTREE_Training)^2)
  MSE_TREE22_Training
  #0.02586462
  
  
#BAGGING using RANDOMFOREST FUNCTION 
  
  randomForest(RATING ~ ., data=dt_training, 
               SingleBaggingtry=(ncol(dt_training)-1), ntree=500)
  # the number of variables tried at each split was 7
  # MS Residuals: 0.02458712, %var explained 69.7
  
  #If the number of predictors is large but the number of trees is too small, then some features can (theoretically) be missed 
  # in all subspace used. 
  #Both cases results in the decrease of random forest predictive power. 
  #But the last is a rather extreme case, since the selection of subspace is performed at each node.
  

  #Random forest uses bagging (picking a sample of observations rather than all of them for each split) 
  
  
  #Using the training dataset
  baggingmodel <- randomForest(RATING ~ ., data=dt_training, 
                            mtry=(ncol(dt_training)-1), ntree=500)
  
  Resultspredictedbagging<- data.table(
    actual = dt_training$RATING, # y
    prediction = predict(baggingmodel, newdata = dt_training))
  
  MSE_FOREST_Training <- mean((dt_training$RATING - Resultspredictedbagging$prediction)^2)
  MSE_FOREST_Training
  #MSE TRAINING 0.00361235
  
  #Using the TEST dataset
  baggingmodeltest <- randomForest(RATING ~ ., data=dt_test, 
                               mtry=(ncol(dt_test)-1), ntree=500)
  
  Resultspredictedbaggingtest<- data.table(
    actual = dt_test$RATING,
    prediction = predict(baggingmodel, newdata = dt_test))
  
  MSE_FOREST_Test <- mean((dt_test$RATING - Resultspredictedbaggingtest$prediction)^2)
  MSE_FOREST_Test
  #MSE TEST 0.03580105
  
#THE FOREST PERFORM BETTER IN MSE TRAINING AND WORST IN THE MSE OF TESTING THIS IS OBVIOUS A LIKELY OVERFITTING
  # TO THE TRAINING DATA SO WE CAN SET THE NUMBER OF TRESS LOWER THAN 1000 (SUITABLE SINCE THE NOBS IS 396) OR/AND
  #MAKE REPEATED CROSSVALIDATION TO GET A BETTER PERFORMANCE INSTEAD OF THE ACTUAL HOLD OUT
  # AND THE LAST OPTION IS TO MAKE REGULARIZATION TO PENALIZE COMPLEX MODELS AND GET GENERALIZED MODEL
  
  
#FIRST WE CAN USE THE TUNERF function to see the behaviour   
  
#TUNNING To set the number of random inputs selected at each split,
  
  randomForest(RATING ~ ., data=dt_training, ntree=500) 
  tuneRF(x=dt_training[,-"RATING"], y=dt_training[,RATING], ntreeTry = 500) 
  #The out-of-bag (OOB) error is the average error for each calculated using predictions 
  #from the trees that do not contain in their respective bootstrap sample. 
  #This allows the RandomForestClassifier to be fit and validated whilst being trained
  
  
#REPEATED CROSS VALIDATION OF RANDOM FOREST
    
  train_control <- trainControl("repeatedcv", number=5, repeats = 10)
  metric <- "Accuracy" # estimation index
  x<-dt_training[,2:23]
  y<-dt_training[,1]
  
  mtry <- sqrt(ncol(x))
  tunegrid <- expand.grid(.mtry=mtry)
  
  
  
  FOREST_repeatedcv <- train(RATING ~ .,
                              data = dt_training,
                              method = "rf",
                              trControl = train_control,
                              tuneGrid = tunegrid,
                              )
  
  FOREST_repeatedcv$results
  
  predictionstrain <- predict(FOREST_repeatedcv, newdata = dt_training)
  MSE_training_forestcv <- mean((dt_training$RATING-predictionstrain)^2)
  MSE_training_forestcv
  
  predictionstest <- predict(FOREST_repeatedcv, newdata = dt_test)
  MSE_test_forestcv <- mean((dt_test$RATING-predictionstest)^2)
  MSE_test_forestcv
  
  results <- FOREST_repeatedcv$results
  
  #__________________________________________________________________________________________________
  
#GRADIENT BOOSTING just one CV

  model_gForest <- gbm(RATING ~ ., 
                   data=dt_training, 
                   distribution="gaussian",
                   interaction.depth=1,
                   n.trees=500, 
                   shrinkage=0.1,
                   cv.folds = 5
  )
  
  
  gbm.perf(model_gForest, method = "cv")
  
  
#ENSAMBLE PERFORMANCE______________________________________________________________________________________
  #enhance the accuracy and gerneralization abilities
  #test error
  
  # Size of ensembles
  ntrees <- 500
  
  # Lists to collect results
  model_bag_rmse <- vector("list", ntrees)
  model_rf_rmse <- vector("list", ntrees)
  model_gbt_rmse <- vector("list", ntrees)
  
  # One-time models
  model_tree <- rpart(RATING ~ ., data=dt_training, method="anova")
  model_tree_rmse <- sqrt(mean((dt_test$RetailPrice - 
                                  predict(model_tree, newdata = dt_test))^2))
  
  model_gbt <- gbm(RATING ~ ., data=dt_training, distribution='gaussian', 
                   n.trees=ntrees, shrinkage=0.05, interaction.depth=1) # depth = 1
  
  # Loop for MSE  
  for(i in 1:ntrees){
    model_bag <- randomForest(RATING ~ ., data=dt_training, 
                              mtry=(ncol(dt_training)-1), ntree=i) 
    model_bag_rmse[[i]] <- sqrt(mean((dt_test$RATING - 
                                        predict(model_bag, newdata = dt_test))^2))
    model_rf <- randomForest(RATING ~ ., data=dt_training, ntree=i) 
    model_rf_rmse[[i]] <- sqrt(mean((dt_test$RATING - 
                                       predict(model_rf, newdata = dt_test))^2)) 
    model_gbt_rmse[[i]] <- sqrt(mean((dt_test$RATING - 
                                        predict(model_gbt, newdata = dt_test, n.trees=i))^2))
  }
  
  # Organize results
  dt_results <- data.table(
    "n" = 1:ntrees, #FROM 1 TO 500 TREES
    "BoostedTrees" = unlist(model_gbt_rmse),
    "RandomForest"= unlist(model_rf_rmse), 
    "Bagging" = unlist(model_bag_rmse), 
    "SingleTree" = rep(model_tree_rmse,ntrees))
  
  # Plot results
  ggplot(data = dt_results, aes(x=n)) +
    geom_line(aes(y=SingleTree), color="#000000", linetype="dashed") +
    geom_line(aes(y=Bagging), color="#4254f5") +
    geom_line(aes(y=RandomForest), color="#f55742") +
    geom_line(aes(y=BoostedTrees), color="#42f55a") +
    labs(
      x = "Number of trees",
      y = "RMSE",
      title="Ensemble Performances") +
    theme_minimal() +
    theme(axis.line = element_line(color = "#000000"),
          axis.ticks=element_blank())
  

#OOB ERROR_________________________________________________________________________________________________
  
  ntrees <- 500
  
  model_rf_rmse_test <- vector("list", ntrees)
  model_rf_rmse_train <- vector("list", ntrees)
  model_rf_rmse_oob <- vector("list", ntrees)
  
  for(i in 1:ntrees){
    model_rf <- randomForest(RATING ~ ., data=dt_training, ntree=i)
    #OOB error
    model_rf_rmse_oob[[i]] <- sqrt(tail(model_rf$mse, 1))
    #test error
    model_rf_rmse_test[[i]] <- sqrt(mean((dt_test$RATING - 
                                            predict(model_rf, newdata = dt_test))^2)) 
    #training error
    model_rf_rmse_train[[i]] <- sqrt(mean((dt_training$RATING - 
                                             predict(model_rf, newdata = dt_training))^2)) 
  }
  
  dt_results <- data.table("n" = 1:ntrees, 
                           "Test" = unlist(model_rf_rmse_test),
                           "Train"= unlist(model_rf_rmse_train), 
                           "OOB" = unlist(model_rf_rmse_oob))
  
  ggplot(data = dt_results, aes(x=n)) +
    geom_line(aes(y=Test), color="#4CA7DE") +
    geom_line(aes(y=Train), color="#f1b147") +
    geom_line(aes(y=OOB), color="#000000", linetype="dashed") +
    labs(
      x = "Number of trees",
      y = "RMSE") +
    theme_minimal() +
    theme(axis.line = element_line(color = "#000000"),
          axis.ticks=element_blank())
  
  
  
#CONCLUSSION In this case after 10, 20 trees the training set remains stabilize and the OOB RMSE IS HIGHER THAN THE TEST RMSE so this indicates issues with model fitting or generalization. 
#In this case the model is becoming too complex and capturing noise in the training data, which leads to worse performance on unseen data (test set).
#The optimal number of trees is the point where the test RMSE is at its lowest and starts to stabilize or slightly increase. Seems to be around 10-20 trees, where the blue line levels off.
 
  
#Repeating everything with 10 trees
  
  ntrees1 <- 10
  
  model_rf_rmse_test20 <- vector("list", ntrees1)
  model_rf_rmse_train20 <- vector("list", ntrees1)
  model_rf_rmse_oob20 <- vector("list", ntrees1)
  
  for(i in 1:ntrees1){
    model_rf20 <- randomForest(RATING ~ ., data=dt_training, ntree=i)
    #OOB error
    model_rf_rmse_oob20[[i]] <- sqrt(tail(model_rf20$mse, 1))
    #test error
    model_rf_rmse_test20[[i]] <- sqrt(mean((dt_test$RATING - 
                                            predict(model_rf20, 
                                                    newdata = dt_test))^2)) 
    #training error
    model_rf_rmse_train20[[i]] <- sqrt(mean((dt_training$RATING - 
                                             predict(model_rf20, 
                                                     newdata = dt_training))^2)) 
  }
  
  dt_results20 <- data.table("n" = 1:ntrees1, 
                           "Test" = unlist(model_rf_rmse_test20),
                           "Train"= unlist(model_rf_rmse_train20), 
                           "OOB" = unlist(model_rf_rmse_oob20))
  
  ggplot(data = dt_results20, aes(x=n)) +
    geom_line(aes(y=Test), color="#4CA7DE") +
    geom_line(aes(y=Train), color="#f1b147") +
    geom_line(aes(y=OOB), color="#000000", linetype="dashed") +
    labs(
      x = "Number of trees",
      y = "RMSE") +
    theme_minimal() +
    theme(axis.line = element_line(color = "#000000"),
          axis.ticks=element_blank())
  
# The difference between OOB error and testing error may be caused by the size 
# of the dataset and the differences in sample distribution.
  
  