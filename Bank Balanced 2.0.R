# Package list
list_of_packages <- c("data.table" #frame with syntax and feature enhancements for ease of use, convenience and programming speed
                      ,"readr" # data import
                      ,"dplyr" # data manipulation
                      ,"tidyverse" # to order the data
                      ,"ggplot2" #graphs
                      ,"rpart" # decision trees
                      ,"rpart.plot" # nice tree's charts
                      ,"randomForest" # random forests
                      ,"caret" #importance
                      ,"ROCR" # model quality assessment - ROC curve, AUC, etc.
                      ,"MASS" # selection of variables for the model
)

# Installing missing libraries
not_installed <- list_of_packages[!(list_of_packages %in% installed.packages()[ , "Package"])]
if(length(not_installed)) install.packages(not_installed)

# Loading libraries
lapply(list_of_packages, library, character = TRUE)



#DATA
rm(list=ls()) # cleaning the environment

bank <- read.csv2("/Users/stefanofiorini/Desktop/bank+marketing/bank/bank-full.csv", stringsAsFactors = FALSE)

str(bank)
apply(bank, 2, unique) # appy to the column

table(bank$poutcome)
table(bank$previous)


#FEATURES ENGiNEERING

##reduntat with the "previous" variable, in order to reduce the risk of overfitting
bank$poutcome <- factor(bank$poutcome, levels = c("failure", "other", "success", "unknown"))
bank$poutcome_num <- as.numeric(bank$poutcome)

cor(bank$previous, bank$poutcome_num)

bank$poutcome <- NULL #
#Both values were moderately correlated
cor(bank$previous, bank$pdays)
bank$pdays <- NULL #reduntat with the "previous" variable, in order to reduce the risk of overfitting

#This variable is removed even though it was a strong predictor based on the following dataset author’s notes.
##Note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
bank$duration <- NULL


#DEALING WITH UNKNOWS

any(is.na(bank))
apply(bank, 2, anyNA)

bank_clean <- bank %>%
  mutate(across(everything(), ~ifelse(tolower(.x) == "unknown", NA, .x)))

any(is.na(bank_clean))
colSums(is.na(bank_clean))

bank_clean <- na.omit(bank_clean)
any(is.na(bank_clean))

bank_clean$y <- factor(bank_clean$y)
str(bank_clean)

#TRAIN/TEST DATA

set.seed(17)
test_prop <- 0.3
test.set.index <- (runif(nrow(bank_clean)) < test_prop)
test <- bank_clean[test.set.index, ]
train <- bank_clean[!test.set.index, ]

#UNDERSAMPLING
min_n <- min(table(train$y))

train_balanced <- train %>%
  group_by(y) %>%
  sample_n(min_n) %>%
  ungroup()


#MODELS BALANCED
#LOGISTIC REGRESSION
log_reg <- glm(y ~ ., data = train_balanced, family = "binomial")

#DECISION TREE
tree <- rpart(y ~ .,
              data = train_balanced,
              method = "class",
              cp = 0.004)

tree
rpart.plot(tree, under = FALSE, tweak = 0.9, fallen.leaves = TRUE)

#RANDOM FOREST
rf <- randomForest(y ~., 
                   data = train_balanced)
plot(rf)

train_control <- trainControl(method = "cv", number = 5)
rf_2 <- train(y ~ ., data = train_balanced, method = "rf", trControl = train_control)
plot(rf_2$finalModel, main = "rf_2 error plot")


#EVALUATION

EvaluateModel <- function(classif_mx){
  true_positive <- classif_mx[2, 2]
  true_negative <- classif_mx[1, 1]
  false_negative <- classif_mx[2, 1]
  condition_positive <- sum(classif_mx[ , 2])
  condition_negative <- sum(classif_mx[ , 1])
  predicted_positive <- sum(classif_mx[2, ])
  predicted_negative <- sum(classif_mx[1, ])
  
  accuracy <- (true_positive + true_negative) / sum(classif_mx)
  MER <- 1 - accuracy # Misclassification Error Rate
  # MER < - (false_positive + false_positive) / sum(classif_mx)
  precision <- true_positive / predicted_positive
  sensitivity <- true_positive / condition_positive #  Recall / True Positive Rate (TPR)
  specificity <- true_negative / condition_negative
  FOR <- false_negative / predicted_negative
  F1 <- (2 * precision * sensitivity) / (precision + sensitivity)
  return(list(accuracy = accuracy, 
              MER = MER,
              precision = precision,
              sensitivity = sensitivity,
              specificity = specificity,
              FOR = FOR,
              F1 = F1))
}

sapply(CM, EvaluateModel)



#EVALUATION - NON BALANCED

CM_1 <- list()
### Trees
CM_1[["log_reg"]] <- table(ifelse(predict(log_reg, new = test, type = "response") > 0.5, 1, 0), test$y)
### Trees
CM_1[["tree"]] <- table(predict(tree, new = test, type = "class"),test$y)
### Random forest
CM_1[["rf"]] <- table(predict(rf, new = test, type = "class"), test$y)
CM_1[["rf_2"]] <- table(predict(rf_2, new = test, type = "raw"), test$y)

sapply(CM_1, EvaluateModel)


preds_1 <- list()

### Regressions
preds_1[["log_reg"]] <- as.vector(predict(log_reg, newdata = test, type = "response"))
### Trees
preds_1[["tree"]] <- as.vector(predict(tree, newdata = test)[, 2])
### Random forest
preds_1[["rf"]] <- as.vector(predict(rf, newdata = test, type = "prob")[, 2])
preds_1[["rf_2"]] <- as.vector(predict(rf_2, newdata = test, type = "prob")[, 2])

## ROC curve (Receiver Operating Characteristic) - needs a "continuous" forecast

for (i in 1:length(preds_1)){
  plot(performance(prediction(preds_1[[i]], test$y), "tpr", "fpr"), lwd = 2, colorize = F, col = i,  add = ifelse(i == 1, FALSE, TRUE), main ="ROC curve - Models 1") 
}

abline(coef = c(0, 1), lty = 2, lwd = 0.5)

legend(0.6, 0.4, 
       legend = names(preds_1),
       col = 1:length(preds_1), 
       lty = rep(1, length(preds_1))
)

# AUC (Area Under Curve) - under ROC curve

for (i in 1:length(preds_1)){
  cat(names(preds_1)[i], ": ", performance(prediction(preds_1[[i]], test$y), "auc")@y.values[[1]], "\n")
}



#MODELS PART 2

#LOGISTIC REGRESSION
weights <- ifelse(train$y == 1, 3, 1)

log_reg_w <- glm(y ~ ., data = train, family = "binomial", weights = weights)


#DECISION TREE ALL DATA
tree_w <- rpart(y ~ ., data = train, method = "class",
                    cp = 0.0024,
                    parms = list(split = "information", prior = c(0.25, 0.75)))  

rpart.plot(tree_w, under = FALSE, tweak = 0.9, fallen.leaves = TRUE)

#DECISION TREE BALANCED DATA
tree_w_balanced <- rpart(y ~ ., data = train_balanced, method = "class",
                         cp = 0.0024,
                         parms = list(split = "information",prior = c(0.7, 0.3)))  

rpart.plot(tree_w_balanced, under = FALSE, tweak = 0.9, fallen.leaves = TRUE)


# RAIN FOREST ALL DATA
install.packages(c("themis", "recipes", "ranger"))
library(themis)
library(recipes)
library(ranger)

# Create a recipe with SMOTE
rec <- recipe(y ~ ., data = train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(y) %>%
  prep()  # Important: prep the recipe

# Apply the recipe to get processed data
train_processed <- juice(rec)  # or bake(rec, new_data = train)

# Alternative: Extract predictors and response separately
X_train <- train_processed %>% dplyr::select(-y)
y_train <- train_processed$y
# Train model with recipe

rf_smote <- caret::train(
  x = X_train,
  y = y_train,
  method = "ranger",
  trControl = train_control,
  tuneGrid = expand.grid(
    mtry = c(2, 4, 6, 8),
    splitrule = "gini",
    min.node.size = c(1, 5, 10)
  )
)

rf_smote$results
rf_smote$bestTune


rf_smote_best <- randomForest(
  y ~ ., 
  data = train,
  ntree = 500,
  mtry = 4,
  nodesize = 5,         # questo è l'equivalente di min.node.size
  importance = TRUE
)

plot(rf_smote_best)


# RAIN FOREST BALANCED DATA

# Create a recipe with SMOTE
rec_balanced <- recipe(y ~ ., data = train_balanced) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(y) %>%
  prep()  # Important: prep the recipe

# Apply the recipe to get processed data
train_processed_bal <- juice(rec_balanced)  # or bake(rec, new_data = train)

# Alternative: Extract predictors and response separately
X_train_bal <- train_processed_bal %>% dplyr::select(-y)
y_train_bal <- train_processed_bal$y
# Train model with recipe

rf_smote_balanced <- caret::train(
  x = X_train_bal,
  y = y_train_bal,
  method = "ranger",
  trControl = train_control,
  tuneGrid = expand.grid(
    mtry = c(2, 4, 6, 8),
    splitrule = "gini",
    min.node.size = c(1, 5, 10)
  )
)

rf_smote_balanced$results
rf_smote_balanced$bestTune


rf_smote_best_balanced <- randomForest(
  y ~ ., 
  data = train_balanced,
  mtry = 6,
  nodesize = 10,         # questo è l'equivalente di min.node.size
  importance = TRUE
)

plot(rf_smote_best_balanced)

#EVALUATION 2
CM_2 <- list()
#Logistic regression
CM_2[["log_reg_w"]] <- table(ifelse(predict(log_reg_w, new = test, type = "response") > 0.5, 1, 0), test$y)
### Trees
CM_2[["tree_w"]] <- table(predict(tree_w, new = test, type = "class"),test$y)
CM_2[["tree_w_balanced"]] <- table(predict(tree_w_balanced, new = test, type = "class"),test$y)

### Random forest
CM_2[["rf_smote_best_balanced"]] <- table(predict(rf_smote_best_balanced, new = test, type = "class"), test$y)

sapply(CM_2, EvaluateModel)


preds_2 <- list()

### Regressions
preds_2[["log_reg_w"]] <- as.vector(predict(log_reg_w, newdata = test, type = "response"))
### Trees
preds_2[["tree_w"]] <- as.vector(predict(tree_w, newdata = test)[, 2])
preds_2[["tree_w_balanced"]] <- as.vector(predict(tree_w_balanced, newdata = test)[, 2])
### Random forest
preds_2[["rf_smote_best_balanced"]] <- as.vector(predict(rf_smote_best_balanced, newdata = test, type = "prob")[, 2])

## ROC curve (Receiver Operating Characteristic) - needs a "continuous" forecast

for (i in 1:length(preds_2)){
  plot(performance(prediction(preds_2[[i]], test$y), "tpr", "fpr"), lwd = 2, colorize = F, col = i,  add = ifelse(i == 1, FALSE, TRUE), main ="ROC curve - Models 2") 
}

abline(coef = c(0, 1), lty = 2, lwd = 0.5)

legend(0.6, 0.4, 
       legend = names(preds_2),
       col = 1:length(preds_2), 
       lty = rep(1, length(preds_2))
)

# AUC (Area Under Curve) - under ROC curve

for (i in 1:length(preds_2)){
  cat(names(preds_2)[i], ": ", performance(prediction(preds_2[[i]], test$y), "auc")@y.values[[1]], "\n")
}










#MODELS PART 3
#DECISION TREE

importance_twb <- tree_w_balanced$variable.importance
print(importance_twb)

threshold <- 0.10 * max(importance_twb)
selected_vars <- names(importance_twb[importance_twb >= threshold])

formula_selected <- as.formula(paste("y ~", paste(selected_vars, collapse = " + ")))
tree_twb <- rpart(formula_selected, data = train_balanced, method = "class", cp = 0.002,
                  parms = list(split = "information",prior = c(0.7, 0.3)))

rpart.plot(tree_twb, under = FALSE, tweak = 0.9, fallen.leaves = TRUE)

#RAIN FOREST LESS
varImp(rf_smote_best_balanced)

rf_smote_best_balanced_less <- randomForest(
  y ~ .- default - job , 
  data = train_balanced,
  mtry = 6,
  nodesize = 10,         # questo è l'equivalente di min.node.size
  importance = TRUE
)


#EVALUATION PART 3
CM_3 <- list()
### Trees
CM_3[["tree_twb"]] <- table(predict(tree_twb, new = test, type = "class"),test$y)
### Random forest
CM_3[["rf_smote_best_balanced_less"]] <- table(predict(rf_smote_best_balanced_less, new = test, type = "class"), test$y)

sapply(CM_1, EvaluateModel)
sapply(CM_2, EvaluateModel)
sapply(CM_3, EvaluateModel)

preds_3 <- list()

#Decision tree
preds_3[["tree_twb"]] <- as.vector(predict(tree_twb, newdata = test)[, 2])
### Random forest
preds_3[["rf_smote_best_balanced_less"]] <- as.vector(predict(rf_smote_best_balanced_less, newdata = test, type = "prob")[, 2])

## ROC curve (Receiver Operating Characteristic) - needs a "continuous" forecast

for (i in 1:length(preds_3)){
  plot(performance(prediction(preds_3[[i]], test$y), "tpr", "fpr"), lwd = 2, colorize = F, col = i,  add = ifelse(i == 1, FALSE, TRUE), main ="ROC curve - Models 3") 
}

abline(coef = c(0, 1), lty = 2, lwd = 0.5)

legend(0.6, 0.4, 
       legend = names(preds_3),
       col = 1:length(preds_3), 
       lty = rep(1, length(preds_3))
)

# AUC (Area Under Curve) - under ROC curve

for (i in 1:length(preds_3)){
  cat(names(preds_3)[i], ": ", performance(prediction(preds_3[[i]], test$y), "auc")@y.values[[1]], "\n")
}


#MODELS PART 4
install.packages("bonsai")
install.packages("parsnip")
install.packages("lightgbm")
install.packages("tune")
install.packages("rlang")
install.packages("dials")
install.packages("tidymodels")
install.packages("finetune")

library(bonsai)
library(parsnip)
library(lightgbm)
library(tune)
library(rlang)
library(dials)
library(tidymodels)
library(finetune)


#iniial split object
min_n_all <- min(table(bank_clean$y))
bank_clean_balanced <- bank_clean %>%
  group_by(y) %>%
  sample_n(min_n_all) %>%
  ungroup() 
bank_clean_balanced$y <- as.character(bank_clean_balanced$y)
bank_clean_split <- initial_split(bank_clean_balanced,
                                  strata = "y")

# Create training data from the split
train_balanced <- training(bank_clean_split)





rec_hyper <- recipe(y ~ ., data = train_balanced) |>
  step_dummy(all_nominal_predictors())  

bonsai_spec <- 
  boost_tree(learn_rate = tune(),
             stop_iter = tune(),
             trees = 1000) |>
  set_engine(engine = "lightgbm",
             num_leaves= tune()) |>
  set_mode("classification")


bonsai_spec |>
  extract_parameter_set_dials()

grid_tune <-
  bonsai_spec |>
  extract_parameter_set_dials() |>
  grid_latin_hypercube(size = 250)

grid_tune |> glimpse (width = 250)

install.packages("doMC")
library(doMC)
registerDoMC (cores = 8)

bonsai_spec_wf <- 
  workflow() |>
  add_recipe(rec_hyper) |>
  add_model(bonsai_spec)

#control for the grid
#cntl <- control_grid(save_pred = TRUE,
#                     save_workflow = TRUE)


race_cntl <- control_race(save_pred = TRUE,
                          save_workflow = TRUE)

bonsai_folds <- vfold_cv(train_balanced, v = 10, strata = y)

install.packages("lme4")
library(lme4)

autoplot(bonsai_tune_fast)


bonsai_best_id <-
  bonsai_tune_fast |>
  select_best(metric = "roc_auc")


#BONSAI BEST TREE - AFTER HYPERPARAMETERS TUNING PT.2
bonsai_best_tree <-
  bonsai_tune_fast |>
  extract_workflow() |>
  finalize_workflow(bonsai_best_id) |>
  last_fit(bank_clean_split)

#collect the metrics for the best model
bonsai_best_tree |>
  collect_metrics()



#RANDOM FOREST
install.packages("mlr")
library(mlr)

train_balanced_mutate <- mutate(train_balanced, 
                      month = factor(month,  levels = c("jan", "feb", "mar", "apr", "may", "jun", 
                                                        "jul", "aug", "sep", "oct", "nov", "dec"), 
                                     ordered = TRUE),
                      job = factor(job, levels = c("management",  "blue-collar" ,  "technician",    "services" ,     "admin." ,  "unemployed",   
                                                   "entrepreneur",  "housemaid",     "retired" ,      "self-employed", "student"),
                                   ordered = TRUE),
                      education = factor(education), 
                      marital = factor(marital),
                      default = factor(default),
                      housing = factor(housing), 
                      loan = factor(loan),
                      contact = factor(contact), 
                      y = factor(y)
)
#create a task
train_task <- makeClassifTask(data = train_balanced_mutate, target = "y")

rf_to_be <- makeLearner("classif.randomForest",
                        predict.type = "response",
                        par.vals = list(ntree = floor(0.1*nrow(train_balanced_mutate)),
                                        mtry = floor((ncol(train_balanced_mutate)-1)/3)))
rf$par.vals <- list(importance = TRUE)                        

#grid search to find hyper parameters (number of hyper parameters we are going to utilize)
rf_param <- makeParamSet(
  makeIntegerParam("ntree", lower = 10, upper = floor(0.1*nrow(train_balanced_mutate))),
  makeIntegerParam("mtry", lower = floor((ncol(train_balanced_mutate)-1)/3),
                   uppert = ncol(train_balanced_mutate)))
                   
rf_cntl <- makeTuneControlRandom(maxit = 10L)

set_rf_cv <- makeResampleDesc("CV", iters = 3L)

rf_tune <- tuneParams(learner = rf_to_be,
                      resampling = set_rf_cv,
                      task = train_task,
                      par.set = rf_param,
                      control = rf_cntl,
                      measure = acc)


best_rf_tune  <- randomForest(
  y ~ . , 
  data = train_balanced,
  ntree = 609,
  mtry = 6,        
  importance = TRUE
)


#EVALUATION PART 4
CM_4 <- list()

#tree's metrics won't be calculated in this case, because in the "bonsai_best_tree |> collect_metrics()" we found that accuracy is 0.726 for this model

### Random forest
CM_4[["best_rf_tune"]] <- table(predict(best_rf_tune, new = test, type = "class"), test$y)
sapply(CM_4, EvaluateModel)


preds_4 <- list()

### Random forest
preds_4[["best_rf_tune"]] <- as.vector(predict(best_rf_tune, newdata = test, type = "prob")[, 2])

## ROC curve (Receiver Operating Characteristic) - needs a "continuous" forecast

for (i in 1:length(preds_4)){
  plot(performance(prediction(preds_4[[i]], test$y), "tpr", "fpr"), lwd = 2, colorize = F, col = i,  add = ifelse(i == 1, FALSE, TRUE), main ="ROC curve - Models 4") 
}

abline(coef = c(0, 1), lty = 2, lwd = 0.5)

legend(0.6, 0.4, 
       legend = names(preds_4),
       col = 1:length(preds_4), 
       lty = rep(1, length(preds_4))
)

# AUC (Area Under Curve) - under ROC curve

for (i in 1:length(preds_4)){
  cat(names(preds_4)[i], ": ", performance(prediction(preds_4[[i]], test$y), "auc")@y.values[[1]], "\n")
}









sapply(CM_1, EvaluateModel)
sapply(CM_2, EvaluateModel)
sapply(CM_3, EvaluateModel)
sapply(CM_4, EvaluateModel)




#Graph Trees
preds_5 <- list()

#Decision tree
preds_5[["tree"]] <- as.vector(predict(tree, newdata = test)[, 2])
preds_5[["tree_w"]] <- as.vector(predict(tree_w, newdata = test)[, 2])
preds_5[["tree_w_balanced"]] <- as.vector(predict(tree_w_balanced, newdata = test)[, 2])
preds_5[["tree_twb"]] <- as.vector(predict(tree_twb, newdata = test)[, 2])

## ROC curve (Receiver Operating Characteristic) - needs a "continuous" forecast

for (i in 1:length(preds_5)){
  plot(performance(prediction(preds_5[[i]], test$y), "tpr", "fpr"), lwd = 2, colorize = F, col = i,  add = ifelse(i == 1, FALSE, TRUE), main ="ROC curve - Classification Trees") 
}

abline(coef = c(0, 1), lty = 2, lwd = 0.5)

legend(0.6, 0.4, 
       legend = names(preds_5),
       col = 1:length(preds_5), 
       lty = rep(1, length(preds_5))
)

# AUC (Area Under Curve) - under ROC curve

for (i in 1:length(preds_5)){
  cat(names(preds_5)[i], ": ", performance(prediction(preds_5[[i]], test$y), "auc")@y.values[[1]], "\n")
}


CM_5 <- list()
CM_5[["tree"]] <- table(predict(tree, new = test, type = "class"),test$y)
CM_5[["tree_w"]] <- table(predict(tree_w, new = test, type = "class"),test$y)
CM_5[["tree_w_balanced"]] <- table(predict(tree_w_balanced, new = test, type = "class"),test$y)
CM_5[["tree_twb"]] <- table(predict(tree_twb, new = test, type = "class"),test$y)

sapply(CM_5, EvaluateModel)








rm(preds_6)
#Graph Trees
preds_6 <- list()

#Decision tree
preds_6[["rf"]] <- as.vector(predict(rf,newdata = test,  type = "prob")[, 2])
preds_6[["rf_2"]] <- as.vector(predict(rf_2,newdata = test, type = "prob")[, 2])
preds_6[["rf_smote_best_balanced"]] <- as.vector(predict(rf_smote_best_balanced, newdata = test, type = "prob")[, 2])
preds_6[["rf_smote_best_balanced_less"]] <- as.vector(predict(rf_smote_best_balanced_less,newdata = test,  type = "prob")[, 2])
preds_6[["best_rf_tune"]] <- as.vector(predict(best_rf_tune, newdata = test,  type = "prob")[, 2])


## ROC curve (Receiver Operating Characteristic) - needs a "continuous" forecast

for (i in 1:length(preds_6)){
  plot(performance(prediction(preds_6[[i]], test$y), "tpr", "fpr"), lwd = 2, colorize = F, col = i,  add = ifelse(i == 1, FALSE, TRUE), main ="ROC curve - Random Forests") 
}

abline(coef = c(0, 1), lty = 2, lwd = 0.5)

legend(0.6, 0.4, 
       legend = names(preds_6),
       col = 1:length(preds_6), 
       lty = rep(1, length(preds_6))
)

# AUC (Area Under Curve) - under ROC curve

for (i in 1:length(preds_6)){
  cat(names(preds_6)[i], ": ", performance(prediction(preds_6[[i]], test$y), "auc")@y.values[[1]], "\n")
}

CM_6 <- list()
CM_6[["rf"]] <- table(predict(rf, new = test, type = "class"),test$y)
CM_6[["rf_2"]] <- table(predict(rf_2, new = test, type = "raw"),test$y)
CM_6[["rf_smote_best_balanced"]] <- table(predict(rf_smote_best_balanced, new = test, type = "class"),test$y)
CM_6[["rf_smote_best_balanced_less"]] <- table(predict(rf_smote_best_balanced_less, new = test, type = "class"),test$y)
CM_6[["best_rf_tune"]] <- table(predict(best_rf_tune, new = test, type = "class"),test$y)

sapply(CM_6, EvaluateModel)



#Last slide

bank.2 <- read.csv2("/Users/stefanofiorini/Desktop/bank+marketing/bank/bank-full.csv", stringsAsFactors = FALSE)

apply(bank.2, 2, unique) # appy to the column

mean(bank.2$duration[bank.2$duration != 0 & bank.2$poutcome == "failure"])

