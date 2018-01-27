library(data.table)
train = read.csv("D:/datasets/SocGen problem2/train.csv", na.strings = '',stringsAsFactors = F)
test = read.csv("D:/datasets/SocGen problem2/test.csv", na.strings = '', stringsAsFactors = F)

library(dplyr)

full = bind_rows(train,test)

feature_classes = sapply(names(full),function(x){class(full[[x]])})
numeric_feats = names(feature_classes[feature_classes == "numeric"])
categorical_feats = names(feature_classes[feature_classes %in% c("character","integer") ])
categorical_feats = categorical_feats[-c(1)]

#EDA
hist(full$num_var_1)
boxplot(full$num_var_1)
quantile(full$num_var_1)
library(moments)
skewness(full$num_var_1)
hist(sqrt(full$num_var_1))
quantile(full$num_var_1)
skewness(sqrt(full[,8]))

#Missing Value treatment

missing_values = colSums(is.na(full))
missing_columns = names(missing_values[missing_values>0])
missing_columns = missing_columns[-c(5)]

#Imputing the missing values in categorical values with the highest occuring class.
which.max(table(full$cat_var_1))
full[is.na(full$cat_var_1),]$cat_var_1 = "gf"

which.max(table(full$cat_var_3))
full[is.na(full$cat_var_3),]$cat_var_3 = "qt"

which.max(table(full$cat_var_6))
full[is.na(full$cat_var_6),]$cat_var_6 = "zs"

which.max(table(full$cat_var_8))
full[is.na(full$cat_var_8),]$cat_var_8 = "dn"

#converting to factors
#for(i in 1:length(categorical_feats))
#{
#  full[,categorical_feats[i]] = as.factor(full[,categorical_feats[i]])
#}

full$target = as.factor(full$target)

#for(i in 27:50)
#{
#  full[,i] = as.factor(full[,i])
#}

#one hot encoding
options(na.action = 'na.pass')
library(data.table)
library(Matrix)
full1 = full[-c(1)]
train_xg = full1[1:nrow(train),]
test_xg = full1[(nrow(train)+1):nrow(full1),]
#sparse_matrix_train = sparse.model.matrix(target~.-1, data = train_xg)
#sparse_matrix_test = sparse.model.matrix(target~.-1, data = test_xg)

sparse_matrix_full = sparse.model.matrix(target~.-1, data = full1)
sparse_train = sparse_matrix_full[1:nrow(train),]
sparse_test = sparse_matrix_full[(nrow(train)+1):nrow(full1),]
output_vector = train[,"target"] == "1"



#Model fitting - xgboost CV
library(xgboost)
dtrain = xgb.DMatrix(sparse_train, label = output_vector)
dtest = xgb.DMatrix(sparse_test)
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.05, gamma=3, max_depth=11, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, eval_metric = 'auc')

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 500, nfold = 5, showsd = T, stratified = T, maximize = F)

mean(xgbcv$evaluation_log$test_auc_mean)
which.max(xgbcv$evaluation_log$test_auc_mean)
#nrounds = 909

xgb1 <- xgb.train(params = params, data = dtrain, nrounds = 500,verbose = 1,maximize = F)
xgbpred <- predict(xgb1,dtest, type = 'response')
hist(xgbpred)

final = data.frame(test$transaction_id,xgbpred)
colnames(final) = c("transaction_id","target")
write.csv(final , "D:/datasets/SocGen problem2/xgboost20.csv", row.names = F)


# accuracy on test data- 73.4%.
# mean auc on test while cross-validation 


#Model fitting - xgboost

library(xgboost)
xg_boost = xgboost(data = dtrain, 
                   label = output_vector,
                   max_depth = 4, 
                   eta = 0.05, 
                   nrounds = 1000,
                   objective = 'binary:logistic', 
                   eval_metric = 'auc')

pred = predict(xg_boost, newdata = dtest) 

final = data.frame(test$transaction_id,pred)
colnames(final) = c("transaction_id","target")
write.csv(final , "D:/datasets/SocGen problem2/xgboost9.csv", row.names = F)


