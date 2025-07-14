rm(list=ls())
library(ggplot2)
library(data.table)
library(caret)
library(factoextra)
install.packages("tensorflow")
install.packages("keras")
library(tensorflow)
library(keras)
use_condaenv("keras-tf", required = T)
require(keras)
set.seed(555)

# OLS and NN model with training dataset (m1 and m2)
# OLS and NN model with test dataset (m1 and m2)
# OLS and NN model with PCA variables (m1_pca and m2_pca)
# OLS and NN model with CV (m1_cv and m2_cv)
# OLS and NN model with repeated CV (m1_repeated and m2_repeated)

dt_sup <- fread(file = "C:\\Users\\lopo\\OneDrive\\桌面\\course\\statistical machine learning\\dataset.csv")
dt_sup <- na.omit(dt_sup)
dt_sup <- dt_sup[,c("RATING","AVGTIME", "DISTCENT", "NUM_COMMENTS", "NEG_COMMENTPRICE", 
                    "POS_REVIEWS", "NEG_REVIEWS", "perc_BH", "COMCENTER", "PARKLOT", 
                    "OPEN6AM", "OPEN7AM", "OPEN8AM", "OPENAFTER8"
)]

training_sample <- dt_sup[, sample(.N, floor(.N*0.8))]
dt_training <- dt_sup[training_sample]
dt_test <- dt_sup[-training_sample]

# split our data 80-20

#-----------------------
# hold out (training and test)

# OLS (m1)
ols <- lm(data = dt_training[,-"id"], RATING ~ .)
MSE_trained_m1 <- mean(ols$residuals^2)
MSE_trained_m1
predictions <- predict(ols, newdata = dt_test)
MSE_testing_m1 <- mean((dt_test$RATING - predictions)^2)
MSE_testing_m1

# NN_ROT (m2)

Y <- dt_training$RATING
X <- as.matrix(dt_training[,c("AVGTIME", "DISTCENT", "NUM_COMMENTS", "NEG_COMMENTPRICE", 
                              "POS_REVIEWS", "NEG_REVIEWS", "perc_BH", "COMCENTER", "PARKLOT", 
                              "OPEN6AM", "OPEN7AM", "OPEN8AM", "OPENAFTER8"
)])

Y_test <- dt_test$RATING
X_test <- as.matrix(dt_test[,c("AVGTIME", "DISTCENT", "NUM_COMMENTS", "NEG_COMMENTPRICE", 
                               "POS_REVIEWS", "NEG_REVIEWS", "perc_BH", "COMCENTER", "PARKLOT", 
                               "OPEN6AM", "OPEN7AM", "OPEN8AM","OPENAFTER8"
)])
# scale from keras network rather than scale()
normalize <- layer_normalization()
normalize %>% adapt(X)
model <- keras_model_sequential()
print(normalize$mean)
model %>% 
  normalize() %>%
  layer_dense(units = 8, activation = 'sigmoid') %>%
  layer_dense(units = 16, activation = 'sigmoid') %>%
  layer_dense(units = 1)

summary(model)

# Adam, an algorithm for first-order gradient-based optimization of stochastic 
# objective functions
# gradient descent (weight and bias)
model %>% compile(
  loss = "MSE",
  optimizer = "adam",
  # metrics = "Accuracy"
)
model_nn <- model %>% fit(
  x=X,
  y=Y,
  epochs = 200,
  verbose=0
)
plot(model_nn)
# blue curve represent the loss of the training data
# black curve represent the loss of the test data

neuronpred <- model %>% predict(x=X,verbose=0)
MSE_trained_m2 <- mean((dt_training$RATING - neuronpred)^2)

neuronpredtest <- model %>% predict(x=X_test,verbose=0)
MSE_testing_m2 <- mean((dt_test$RATING - neuronpredtest)^2)

MSE_trained_m2
MSE_testing_m2

#--------------------------
# CV

# the function to draw random sample and replacement 
split_K <- function(dt, folds){
  dt$id <- ceiling(sample(1:nrow(dt), replace = FALSE, nrow(dt)) / 
                     (nrow(dt) / folds))
  return(dt)
}

# 5 folds
K <- 5 

dt_training <- split_K(dt_training, K)
train_control_cv <- trainControl(method = "cv", number = 5)

# OLS (m1_cv)
lm_caret <- train(
  RATING  ~ .,
  data = dt_training[,-"id"],
  method = "lm",
  trControl = train_control_cv)
summary(lm_caret)
lm_caret$results
predictions <- predict(lm_caret, newdata = dt_training)
MSE_training_m1 <- mean((dt_training$RATING-predictions)^2)
MSE_training_m1
predictions <- predict(lm_caret, newdata = dt_test)
MSE_test_m1 <- mean((dt_test$RATING-predictions)^2)
MSE_test_m1

# neuron network (m2_cv)
library(neuralnet)
# "relu", "tanh", "sigmoid"
# Sigmoid function: maps the input to the range (0, 1)
# and can be used to interpret the output as probabilities.
# In this case, the output values can be transformed by applying the Sigmoid function
# mapping them to the range (0, 1), and then scaling them to the desired range of 1 to 5.
# Tanh function: maps the input to the range (-1, 1) and has a symmetric shape. 
# Similar to the Sigmoid function, the Tanh function can be used to map the output values to the range of 1 to 5. 
# However, the Relu function might not be the optimal choice for mapping the output to a rating range of 1 to 5.
# This is because the Relu function returns 0 for negative inputs, which would result in values below 1 
# and cannot effectively map to the desired rating range of 1 to5.

# default = 10000 epoch
param_grid <- expand.grid(layer1 = c(8),
                          layer2 = c(16),
                          layer3 = 0)

# default act.fct = "logistic"
nn_caret <- train(RATING ~ .,
                  data = dt_training[, -"id"],
                  method = "neuralnet",
                  trControl = train_control_cv,
                  tuneGrid = param_grid,
                  preProcess = "scale"
)
summary(nn_caret)
nn_caret$results

predictions <- predict(nn_caret, newdata = dt_training)
MSE_training_m2 <- mean((dt_training$RATING-predictions)^2)
MSE_training_m2
predictions <- predict(nn_caret, newdata = dt_test)
MSE_test_m2 <- mean((dt_test$RATING-predictions)^2)
MSE_test_m2

#--------------------------------
# pca, threshold = 80%

train_control_pca <- trainControl("cv", number=5,
                              preProcOptions = list(thresh = 0.8))

# OLS (m1_pca)
m1_pca <- train(
  RATING  ~ .,
  data = dt_training[,-"id"],
  method = "lm",
  trControl = train_control_pca,
  preProcess = "pca"
)

summary(m1_pca)
m1_pca$results

predictions <- predict(m1_pca, newdata = dt_training)
MSE_training_m1_pca <- mean((dt_training$RATING-predictions)^2)
MSE_training_m1_pca
predictions <- predict(m1_pca, newdata = dt_test)
MSE_test_m1_pca <- mean((dt_test$RATING-predictions)^2)
MSE_test_m1_pca

# Neuron (m2_pca)
m2_pca <- train(RATING ~ .,
                  data = dt_training[, -"id"],
                  method = "neuralnet",
                  trControl = train_control_pca,
                  tuneGrid = param_grid,
                  preProcess = c("scale", "pca")
)
m2_pca$results

predictions <- predict(m2_pca, newdata = dt_training)
MSE_training_m2_pca <- mean((dt_training$RATING-predictions)^2)
MSE_training_m2_pca
predictions <- predict(m2_pca, newdata = dt_test)
MSE_test_m2_pca <- mean((dt_test$RATING-predictions)^2)
MSE_test_m2_pca

#------------------------
# repeated CV

train_control_repeated <- trainControl("repeatedcv", number=5, repeats = 10)

# OLS (m1_repeatedcv)
m1_repeatedcv <- train(
  RATING  ~ .,
  data = dt_training[,-"id"],
  method = "lm",
  trControl = train_control_repeated
)

summary(m1_repeatedcv)
m1_repeatedcv$results

predictions <- predict(m1_repeatedcv, newdata = dt_training)
MSE_training_m1_repeatedcv <- mean((dt_training$RATING-predictions)^2)
MSE_training_m1_repeatedcv
predictions <- predict(m1_repeatedcv, newdata = dt_test)
MSE_test_m1_repeatedcv <- mean((dt_test$RATING-predictions)^2)
MSE_test_m1_repeatedcv

# Neuron (m2_repeatedcv)
m2_repeatedcv <- train(RATING ~ .,
                  data = dt_training[, -"id"],
                  method = "neuralnet",
                  trControl = train_control_repeated,
                  tuneGrid = param_grid,
                  preProcess = "scale"
)
m2_repeatedcv$results

predictions <- predict(m2_repeatedcv, newdata = dt_training)
MSE_training_m2_repeatedcv <- mean((dt_training$RATING-predictions)^2)
MSE_training_m2_repeatedcv
predictions <- predict(m2_repeatedcv, newdata = dt_test)
MSE_test_m2_repeatedcv <- mean((dt_test$RATING-predictions)^2)
MSE_test_m2_repeatedcv
