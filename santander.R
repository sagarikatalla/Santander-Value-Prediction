setwd("E:/1/git/kaggle")
train = read.csv("train.csv/train.csv")
test = read.csv("test.csv/test.csv")

str(train[,1:10])
View(head(train))

#check missing values
sum(is.na(train))

# train_num = data.frame(apply(train,2,as.numeric))
# train_num$ID = train$ID
# train = train_num

dim(train)
dim(test)

##check duplicate columns
train = train[,colnames(unique(as.matrix(train),MARGIN=2))] 
dim(train)

#kolmogorov smirnov test
# null hypothesis that x and y, same continuous distribution  
df_ks.test = data.frame(names(train)[1],1)
names(df_ks.test) = c("name","p-value")
for(i in 3:length(names(train))){
  ks_test = ks.test(train[,names(train)[i]],test[,names(train)[i]]) 
  df2 = data.frame(names(train)[i],ks_test$p.value)
  names(df2) = c("name","p-value")
  df_ks.test = rbind.data.frame(df_ks.test,df2)
  
}
df_ks.test = df_ks.test[-1,]

##removing variables with p<0.05, reject H0,
# variable distribution different for train and test
remove_cols = df_ks.test$name[df_ks.test$`p-value`<0.05]
train = train[,!(names(train) %in% remove_cols)]

##EDA
#####check correlation#########
#retaining variables highly correlated with dependent variable, cor>0.07
cor_table = cor((train[,-1]))
cor_target = data.frame(cor_table[,1])
cor_target$cor_table...1. = round(cor_target$cor_table...1.,3)
cor_target= subset(cor_target,abs(cor_target$cor_table...1.)!=0)
cor_target= subset(cor_target,abs(cor_target$cor_table...1.)>0.004)
cor_target= subset(cor_target,abs(cor_target$cor_table...1.)>0.03)


#subsetting those columns only
train_sub2 =  train[,names(train) %in% rownames(cor_target)]


##retain non-sparse/non-zero columns
# sparse_columns = data.frame(count = apply(train[,-c(1,2)],2,function(x){table(x)["0"]}))
# sparse_columns$prop = sparse_columns$count*100/nrow(train)
# View((sparse_columns))
# 
# sparse_columns_less_90 = data.frame(sparse_columns[(sparse_columns$prop<90),])
# retain_columns = rownames(sparse_columns_less_90)

# train_sub = train[,retain_columns]
# train_sub$ID = train$ID
# train_sub$target = train$target

##check proportion of zeros, remove 
##drop different distribution columns from test and train

################# FEATURE SELECTION ####################
####### RF REGRESSOR ###########

###### BORUTA VARIMP ##########
set.seed(123)
library(Boruta)
boruta.train <- Boruta((target)~., data=train_sub[,names(train_sub)!="ID"], doTrace = 2)
print(boruta.train)

final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

retain_features = getSelectedAttributes(final.boruta, withTentative = F)

boruta.df <- attStats(final.boruta)
class(boruta.df)
print(boruta.df)


train_sub2=train[,names(train) %in% retain_features]
train_sub2$ID = train$ID
train_sub2$target = train$target
################# FEATURE ENGINEERING ################


#split into train1 and validation
index= sample(1:nrow(train_sub),round(0.8*nrow(train_sub)))
train_sub =train_sub[,-377]
train1 = train_sub[index,]
validation1 = train_sub[-index,]

############ lm ##################
lm_model = lm(target~. ,train_sub)
# lm_model2 = step(lm_model)
summary(lm_model)

lm_model2 = lm(target~. ,train1)
summary(lm_model2)

#prediction
pred = predict(lm_model,validation1[,-1] )

########### svm ##################
library(e1071)
svm_model = svm(target~. ,train_sub2[,names(train_sub2)!="ID"])
# lm_model2 = step(lm_model)
summary(svm_model)

svm_model2 = svm(target~. ,train1)

#prediction
pred_validn_svm = predict(svm_model2,validation1[,-1] )

pred_validn_svm = ifelse(pred_validn_svm<0,0,pred_validn_svm)

##check accuracy on validation dataset
library(forecast)
library(ModelMetrics)
acc_svm = accuracy(pred_validn_svm,validation1$target)
acc_svm

############# random forest  #############  
library(randomForest)
rf_model = randomForest(target~. ,train_sub2[,names(train_sub2)!="ID"])
# lm_model2 = step(lm_model)
summary(rf_model)

#prediction
pred_validn_rf = predict(rf_model,validation1[,-1] )

pred_validn_rf = ifelse(pred_validn_rf<0,0,pred_validn_rf)
pred_validn_rf = (pred_validn_rf+pred_validn_svm)/2
##check accuracy on validation dataset
library(forecast)
acc_rf = accuracy(pred_validn_rf,validation1$target)
acc_rf

############# lasso   #############  
library(randomForest)
rf_model = randomForest(target~. ,train_sub2)
# lm_model2 = step(lm_model)
summary(rf_model)

#prediction
pred_validn_rf = predict(rf_model,validation1[,-1] )

pred_validn_rf = ifelse(pred_validn_rf<0,0,pred_validn_rf)

##check accuracy on validation dataset
library(forecast)
acc_rf = accuracy(pred_validn_rf,validation1$target)
############# ridge  #############  

############# decision tree  #############  

############# neural network  #############  

############# xgboost  #############  
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

labels_train = train_sub2$target
df_train = train_sub2[-grep('target' , colnames(train_sub2))]
df_train = train_sub2[-grep( 'ID', colnames(train_sub2))]

validation2 = validation1[,names(train1) %in% rownames(cor_target)]
labels_test = validation2$target
df_test = test[-grep('target', colnames(train_sub))]

train_mat<- as.matrix(df_train, rownames.force=NA)
test_mat<- as.matrix(df_test, rownames.force=NA)
train_mat <- as(train_mat, "sparseMatrix")
test_mat <- as(test_mat, "sparseMatrix")
# Never forget to exclude objective variable in 'data option'
train_Data <- xgb.DMatrix(data = train_mat, label = labels_train )

# Tuning the parameters #
# Create Empty List
All_rmse<- c()
Param_group<-c()
for (iter in 1:20) {
  param <- list(objective = "reg:linear",
                eval_metric = "rmse",
                booster = "gbtree",
                max_depth = sample(6:10, 1),
                eta = runif(1, 0.01, 0.3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, 0.6, 0.9),
                colsample_bytree = runif(1, 0.5, 0.8)
                
  )
  cv.nround = 500
  cv.nfold = 4
  mdcv <- xgb.cv(data=train_Data, params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,verbose = TRUE)
  # Least Mean_Test_RMSE as Indicator # 
  min_rmse<- min(mdcv$evaluation_log[,"test_rmse_mean"])
  All_rmse<-append(All_rmse,min_rmse)
  Param_group<-append(Param_group,param)
  # Select Param
  param<-Param_group[(which.min(All_rmse)*8+1):(which.min(All_rmse)*8+8)]
}

param<-list(
  objective = "reg:linear",
  eval_metric = "rmse",
  booster = "gbtree",
  max_depth = 22,
  eta = 0.1983,
  gamma = 1.45, 
  subsample = 0.6812,
  colsample_bytree = 0.054
)


#revised boruta


Training <-
  xgb.train(params = param,
            data = train_Data,
            nrounds = 600,
            watchlist = list(train = train_Data),
            verbose = TRUE,
            print_every_n = 50,
            nthread = 6)

test_data <- xgb.DMatrix(data = test_mat,label=labels_test)

prediction <- predict(Training, test_data)
rmse(log(labels_test),log(prediction))

acc_xgb = accuracy(log(prediction),log(labels_test))
rmsle = sqrt(sum((log(prediction+1)-log(labels_test+1))^2)/length(labels_test))
rmsle 

test_mat<- as.matrix(test, rownames.force=NA)
test_mat <- as(test_mat, "sparseMatrix")

############# gbm  #############  
library(gbm)

gbm_model <- gbm(target~. ,
         data=train_sub2[,names(train_sub2)!="ID"],
         distribution = "gaussian",
         interaction.depth=3,
         bag.fraction=0.7,
         n.trees = 10000)

pred_test <- predict(gbm_model,newdata = test, n.trees = 10000)
############# deep learning  #############  



##prediction on test for submission ####
final_model = rf_model
test_data <- xgb.DMatrix(data = test)
#xgb
pred_test <- predict(Training, newdata = as.matrix(test[,-1]))
pred_gbm = predict(gbm_model,newdata = test, n.trees = 10000)
pred_rf = predict(rf_model,newdata = test)
pred_svm = predict(svm_model,newdata = test)

pred_svm1 = predict(svm_model,newdata = test[1:10000,])
pred_svm2 = predict(svm_model,newdata = test[10001:20000,])
pred_svm3 = predict(svm_model,newdata = test[20001:30000,])
pred_svm4 = predict(svm_model,newdata = test[30001:40000,])
pred_svm5 = predict(svm_model,newdata = test[40001:49342,])
pred_svm = c(pred_svm1,pred_svm2,pred_svm3,pred_svm4,pred_svm5)

pred_test = abs(pred_svm)
pred_test = abs(pred_rf) 


pred_test = (abs(pred_rf)+ abs(pred_svm))/2
pred_test = ifelse(pred_test<0,0,pred_test)
pred_test = abs(pred_test)
## submisssion csv ####
sample_submit = read.csv(  "sample_submission.csv/sample_submission.csv")
sample_submit$target = pred_test
write.csv(sample_submit,"submission.csv", row.names=FALSE)

