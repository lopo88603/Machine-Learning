library(data.table)
library(caret)

dt_sup <- fread(file = "C:/Users/Celra/OneDrive/Escritorio/HALLE DATA SCIENCE/Second Semester/3 Statiscal Machine Learning/dataset.csv")

dt_sup <- na.omit(dt_sup)
dt_sup2 <- dt_sup[,c("RATING","AVGTIME", "DISTCENT", "NUM_COMMENTS", "NEG_COMMENTPRICE", 
                     "POS_REVIEWS", "NEG_REVIEWS", "perc_BH", "COMCENTER", "PARKLOT", 
                     "OPEN6AM", "OPEN7AM", "OPEN8AM", "OPENAFTER8")]

training_sample <- dt_sup2[, sample(.N, floor(.N*0.8))]
dt_training2 <- dt_sup2[training_sample]
dt_test2 <- dt_sup2[-training_sample]

train_control <- trainControl("repeatedcv", number=5, repeats = 10)
param_grid <- expand.grid(layer1 = c(51),
                          layer2 = c(13),
                          layer3 = 0)

nnhuang_repeatedcv <- train(RATING ~ .,
                            data = dt_training2[, -"id"],
                            method = "neuralnet",
                            learningrate=0.005,
                            trControl = train_control,
                            tuneGrid = param_grid,
                            preProcess = "scale"
                            )

nnhuang_repeatedcv$results

predictionstrain <- predict(nnhuang_repeatedcv, newdata = dt_training2)
MSE_training_HUANGcv <- mean((dt_training2$RATING-predictionstrain)^2)
MSE_training_HUANGcv

predictionstest <- predict(nnhuang_repeatedcv, newdata = dt_test2)
MSE_test_HUANGcv <- mean((dt_test2$RATING-predictionstest)^2)
MSE_test_HUANGcv


#TUNNING WITH 20 REPETITIONS ON THE REPEATED CROSS VALIDATION

train_control2 <- trainControl("repeatedcv", number=5, repeats = 20)
param_grid2 <- expand.grid(layer1 = c(51),
                          layer2 = c(13),
                          layer3 = 0)

nnhuang_repeatedcv2 <- train(RATING ~ .,
                            data = dt_training2[, -"id"],
                            method = "neuralnet",
                            learningrate=0.005,
                            trControl = train_control2,
                            tuneGrid = param_grid2,
                            preProcess = "scale"
)

predictionstrain2 <- predict(nnhuang_repeatedcv2, newdata = dt_training2)
MSE_training_HUANGcv2 <- mean((dt_training2$RATING-predictionstrain2)^2)
MSE_training_HUANGcv2
#MSETRAINING WITH CV REPETITIONS AT 20: 0.0002299608

predictionstest2 <- predict(nnhuang_repeatedcv2, newdata = dt_test2)
MSE_test_HUANGcv2 <- mean((dt_test2$RATING-predictionstest2)^2)
MSE_test_HUANGcv2
#MSETESTING WITH CV REPETITIONS AT 20:0.038393907

#______________________________________________________________________________________________________

#FINDING IN GRID FOR BETTER RESULTS...........

train_control3 <- trainControl("repeatedcv", number=5, repeats = 20)
param_grid3 <- expand.grid(layer1 = c(10, 20, 30, 40, 50),
                           layer2 = c(5, 10, 15, 20),
                           layer3 = c(0))

nnhuang_repeatedcv3 <- train(RATING ~ .,
                             data = dt_training2[, -"id"],
                             method = "neuralnet",
                             learningrate=0.005,
                             trControl = train_control3,
                             tuneGrid = param_grid3,
                             preProcess = "scale"
)

predictionstrain3 <- predict(nnhuang_repeatedcv3, newdata = dt_training2)
MSE_training_HUANGcv3 <- mean((dt_training2$RATING-predictionstrain3)^2)
MSE_training_HUANGcv3
#MSETRAINING WITH CV REPETITIONS AT 20: 0.0002460

predictionstest3 <- predict(nnhuang_repeatedcv3, newdata = dt_test2)
MSE_test_HUANGcv3 <- mean((dt_test2$RATING-predictionstest3)^2)
MSE_test_HUANGcv3
#MSETESTING WITH CV REPETITIONS AT 20: 0.04840631

#Access grid results

grid_search_results <- nnhuang_repeatedcv3$results
print(grid_search_results)

#_______________________________________________________________________________________________________

#TRYING THE APPROACH OF HIDDEN LAYER 1 (10), HIDDEN LAYER 2 (20)...........


train_control4 <- trainControl("repeatedcv", number=5, repeats = 20)
param_grid4 <- expand.grid(layer1 = c(10),
                           layer2 = c(20),
                           layer3 = 0)

nnhuang_repeatedcv4 <- train(RATING ~ .,
                             data = dt_training2[, -"id"],
                             method = "neuralnet",
                             learningrate=0.005,
                             trControl = train_control4,
                             tuneGrid = param_grid4)

predictionstrain4 <- predict(nnhuang_repeatedcv4, newdata = dt_training2)
MSE_training_HUANGcv4 <- mean((dt_training2$RATING-predictionstrain4)^2)
MSE_training_HUANGcv4
#MSETRAINING : 0.0032942858 

predictionstest4 <- predict(nnhuang_repeatedcv4, newdata = dt_test2)
MSE_test_HUANGcv4 <- mean((dt_test2$RATING-predictionstest4)^2)
MSE_test_HUANGcv4
#MSETESTING : 0.972983
#THE MSE TESTING IS HUGE SO THIS NEURON IS OVERFITTING AND REMEMBERING THE NOISE

#_____________________________________________________________________________________________________

#ADDING ONE MORE LAYER TRYING THE APPROACH OF HIDDEN LAYER1 (10),HIDDEN LAYER2 (20),HL3(30)...........


train_control5 <- trainControl("repeatedcv", number=5, repeats = 20)
param_grid5 <- expand.grid(layer1 = c(10),
                           layer2 = c(20),
                           layer3 = c(30),
                           layer4 = 0)

nnhuang_repeatedcv5 <- train(RATING ~ .,
                             data = dt_training2[, -"id"],
                             method = "neuralnet",
                             learningrate=0.005,
                             trControl = train_control5,
                             tuneGrid = param_grid5)

predictionstrain5 <- predict(nnhuang_repeatedcv5, newdata = dt_training2)
MSE_training_HUANGcv5 <- mean((dt_training2$RATING-predictionstrain5)^2)
MSE_training_HUANGcv5
#MSETRAINING : 0.001637373492

predictionstest5 <- predict(nnhuang_repeatedcv5, newdata = dt_test2)
MSE_test_HUANGcv5 <- mean((dt_test2$RATING-predictionstest5)^2)
MSE_test_HUANGcv5
#MSETESTING : 0.017126789


#CONSTRUCTING LASSO_________________________________________________________________________________________


install.packages("glmnet")
install.packages("glmnetUtils")
library(glmnet)
library(glmnetUtils)


Y <- dt_training2$RATING
X <- as.matrix(dt_training2[,c("AVGTIME", "DISTCENT", "NUM_COMMENTS", "NEG_COMMENTPRICE", 
                              "POS_REVIEWS", "NEG_REVIEWS", "perc_BH", "COMCENTER", "PARKLOT", 
                              "OPEN6AM", "OPEN7AM", "OPEN8AM", "OPENAFTER8")])
Y_test <- dt_test2$RATING
X_test <- as.matrix(dt_test2[,c("AVGTIME", "DISTCENT", "NUM_COMMENTS", "NEG_COMMENTPRICE", 
                               "POS_REVIEWS", "NEG_REVIEWS", "perc_BH", "COMCENTER", "PARKLOT", 
                               "OPEN6AM", "OPEN7AM", "OPEN8AM","OPENAFTER8")])

#Setting a grid search to find the most efficient lambda

lambda<- 10^seq(-3,3, by=0.1)
alpha_seq<-seq(0,1, by=0.1)

cvresult<-cv.glmnet(X,Y, lambda=lambda, alpha=1, nfolds=5) #Alpha 1 because we are using lasso, between 0-1 is L2 
optlambda<- cvresult$lambda.min
optalpha<-cvresult$alpha.min

#After getting the lambda and alpha minimum we replace it and created the lambda model using glmnet
final_model<-glmnet(X,Y,alpha=1, lambda=optlambda)

#To check the predictions
predictionsLASSOTRAIN <- predict(final_model, newx = X)
MSE_LASSOTRAIN <- mean((Y - predictionsLASSOTRAIN)^2)
MSE_LASSOTRAIN
# MSETRAIN LASSO=0.04446234

# To check the predictions for the test set
predictionsLASSOTEST <- predict(final_model, newx = X_test)
MSE_LASSOTEST <- mean((Y_test - predictionsLASSOTEST)^2)
MSE_LASSOTEST
#MSETEST LASSO= 0.03930365

#LOWER MSE TEST THAN MSE TRAIN SO LASSO IS NOT OVERFITTING
#LASSO CAN BE USEFUL AS A REGULARIZED REGRESSION FOR USING A HIGHDIMENSIONAL DATASET WITH SMALL OBS
#LASSO IS HELPING IN PUSHING THE COEFFICIENTS OF LESS IMPORATANT VARIABLES TOWARDS ZERO
#THE RESULT DONT CONSIDER REPEATED CROSSVALIDATION

