#Huang Approach Neural Network

rm(list=ls())
library(ggplot2)
library(factoextra)
library(tensorflow)
library(keras)
require(keras)
library(data.table)

dt_sup <- fread(file = "C:/Users/Celra/OneDrive/Escritorio/HALLE DATA SCIENCE/Second Semester/3 Statiscal Machine Learning/dataset.csv")
dt_sup <- na.omit(dt_sup)

#Seeding a seed for possible replication (unless the result may varying since NN works like that)
set.seed(551)

#SPLITTING THE DATA IN TRAIN SER AND TEST SET
training_sample <- dt_sup[, sample(.N, floor(.N*0.8))]
dt_training <- dt_sup[training_sample]
dt_test <- dt_sup[-training_sample]


Y <- dt_training$RATING
X <- as.matrix(dt_training[,c("AVGTIME", "DISTCENT", "NUM_COMMENTS", "NEG_COMMENTPRICE", 
                         "POS_REVIEWS", "NEG_REVIEWS", "perc_BH", "COMCENTER", "PARKLOT", 
                         "OPEN6AM", "OPEN7AM", "OPEN8AM", "OPENAFTER8")])


Y_test <- dt_test$RATING
X_test <- as.matrix(dt_test[,c("AVGTIME", "DISTCENT", "NUM_COMMENTS", "NEG_COMMENTPRICE", 
                               "POS_REVIEWS", "NEG_REVIEWS", "perc_BH", "COMCENTER", "PARKLOT", 
                               "OPEN6AM", "OPEN7AM", "OPEN8AM","OPENAFTER8")])


# scale from keras network rather than scale()
normalize <- layer_normalization()
Y
normalize %>% adapt(X)
model <- keras_model_sequential()
print(normalize$mean)

# Architecture
model <- keras_model_sequential()

#Huang 2003:  proved that in 2 hidden-layer case, with m output neurons, 
# the number of hidden nodes that are enough to learn N samples with negligibly small error is given by 
# 2???(m+2)N The number of hidden nodes in the first layer is ???(m+2)N + ???N/(m+2)   
#and the second m ???N/(m+2)

#number of hidden nodes= 

model %>% 
  normalize() %>%
  layer_dense(units = 51, activation = 'relu') %>%
  layer_dense(units = 13, activation = 'relu') %>%
  layer_dense(units = 1)

summary(model)
model %>% compile(
  loss = "MSE",
  optimizer = "adam",
  #metrics = "Accuracy"
)

# Train the model with pruning
model_nnHuang <- model %>% fit(
  x=X,
  y=Y,
  epochs = 200, 
  verbose=0,
)


plot(model_nnHuang)
# blue curve represent the loss of the training data
# black curve represent the loss of the test data

model$layers[[2]]$kernel

#PREDICTING

#USING THE TRAINED MODEL WITH THE SAME DATA TRAINING

neuronpredtrain <- model %>% predict(x=X,verbose=0)
# MSE
MSE_neurontrain <- mean((dt_training$RATING - neuronpredtrain)^2)
MSE_neurontrain
# model %>% evaluate(X, Y, verbose=0)

#USING THE TRAINED MODEL WITH THE DATA SET TEST

neuronpredtest <- model %>% predict(x=X_test,verbose=0)
# MSE
MSE_neurontest <- mean((Y_test - neuronpredtest)^2)
MSE_neurontest
# model %>% evaluate(X, Y, verbose=0)

#WITH DATA ON THE TRAINING THE MSE is 0.01267 but with the testing data that 
#the model never saw the MSE is 0.042337

#HUANG NEURAL NETWORK WITH REPEATED CROSS VALIDATION____________________________________________________

library(caret)
dt_sup <- na.omit(dt_sup)
dt_sup2 <- dt_sup[,c("RATING","AVGTIME", "DISTCENT", "NUM_COMMENTS", "NEG_COMMENTPRICE", 
                    "POS_REVIEWS", "NEG_REVIEWS", "perc_BH", "COMCENTER", "PARKLOT", 
                    "OPEN6AM", "OPEN7AM", "OPEN8AM", "OPENAFTER8"
)]

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
                       preProcess = "scale",
                       stepmax=150)

nnhuang_repeatedcv$results

predictions <- predict(nnhuang_repeatedcv, newdata = dt_training2)
MSE_training_HUANGcv <- mean((dt_training$RATING-predictions)^2)
MSE_training_HUANGcv

predictions <- predict(nnhuang_repeatedcv, newdata = dt_test2)
MSE_test_HUANGcv <- mean((dt_test$RATING-predictions)^2)
MSE_test_HUANGcv


#HUANG NEURAL NETWORK WITH RCV (diminish nodes)____________________________________________________

train_control <- trainControl("repeatedcv", number=5, repeats = 10)
param_grid <- expand.grid(layer1 = c(25),
                          layer2 = c(50),
                          layer3 = c(100)
                          )

nnhuang_repeatedcv <- train(RATING ~ .,
                            data = dt_training2[, -"id"],
                            method = "neuralnet",
                            learningrate=0.005,
                            trControl = train_control,
                            tuneGrid = param_grid,
                            preProcess = "scale",
                            stepmax=50)

nnhuang_repeatedcv$results

predictions <- predict(nnhuang_repeatedcv, newdata = dt_training2)
MSE_training_HUANGcv <- mean((dt_training$RATING-predictions)^2)
MSE_training_HUANGcv

predictions <- predict(nnhuang_repeatedcv, newdata = dt_test2)
MSE_test_HUANGcv <- mean((dt_test$RATING-predictions)^2)
MSE_test_HUANGcv


