#!/usr/bin/python
# coding=utf-8

import os
from numpy import *
from sklearn.cross_validation import train_test_split
import pandas as pd
from pandas import *
from pandas import DataFrame
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn import metrics
import time

os.chdir(r'E:\PycharmProjects\Rong360\dta')



def parameter_optimize(dtrain ,dtest,i):

    param = {
        'booster':'gbtree',
	'objective': 'binary:logistic',
        'eval_metric': 'auc',
        
	'gamma':15,
        'eta': 0.045,
        'lambda':17,
        'subsample':0.89,
        'colsample_bytree':0.89,
        'max_depth':9,
        'scale_pos_weight': 1.2,
        
        'missing':-1,
	'seed':520,
	'nthread':4
        }
    num_round = 46
    bst = xgb.train(param, dtrain, num_round)
    
    preds = bst.predict(dtest)
    N_pred = DataFrame(preds,columns=['probability'])
    N_pred['user_id'] = test['user_id']
    N_pred = DataFrame(N_pred,columns=['user_id','probability'])
    N_pred.to_csv('Submit_forecast_dta\\result_1007\\xgb_%s_%s_%s.csv'\
                  %(time.localtime()[1],time.localtime()[2],i),index=False)

    
    bst.save_model('Submit_forecast_dta\\result_1007\\xgb_%s.model'%(i))
    
    return None

def auc_calculate(Mat_Train,Mat_Labels,dtest):
    
    kf = KFold(len(Mat_Train), n_folds=10)
    gather_auc = [];i = 0
    for train_index, test_index in kf:
        print i
        if i > -1:
            X_train, X_test = Mat_Train[train_index], Mat_Train[test_index]
            y_train, y_test = Mat_Labels[train_index], Mat_Labels[test_index]
            dtrain = xgb.DMatrix(X_train, y_train)
            parameter_optimize(dtrain ,dtest,i)
        i = i+1
        
    return None




# 训练集
S_train_user_info = pd.read_csv(r'Generate_dta\S_train_user_info.csv')
N_train_user_info = pd.read_csv(r'Generate_dta\N_train_user_info.csv').drop(['lable'],axis=1)
relation1_train = pd.read_csv(r'Generate_dta\0909relation1_train.csv')
relation2_train = pd.read_csv(r'Generate_dta\0909relation2_train.csv')
N_train_consumption1 = pd.read_csv(r'Generate_dta\N_train_consumption1.csv').drop(['lable'],axis=1)
t_consumption = pd.read_csv(r'Generate_dta\t_consumption.csv')
N_rong_tag_train = pd.read_csv(r'Generate_dta\N_rong_tag_train.csv').drop(['lable'],axis=1)

train = merge(S_train_user_info,N_train_user_info,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation1_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation2_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,N_train_consumption1,how="left", left_on='user_id', right_on='user_id')
train = merge(train,t_consumption,how="left", left_on='user_id', right_on='user_id')

train = train.replace([None], [-1])
train['category_null'] = (train<0).sum(axis=1)
train = train[train['category_null'] < 187]
train = DataFrame(train.values,columns=train.columns)


Mat_Train = train.drop(['user_id','lable'],axis=1)
Mat_Train = array(Mat_Train)

Mat_Labels = train['lable'].astype(int)


# 测试集数据
    # 测试集
S_test_user_info = pd.read_csv(r'Generate_dta\S_test_user_info.csv')
N_test_user_info = pd.read_csv(r'Generate_dta\N_test_user_info.csv')
relation1_test = pd.read_csv(r'Generate_dta\0909relation1_test.csv')
relation2_test = pd.read_csv(r'Generate_dta\0909relation2_test.csv')
N_test_consumption1 = pd.read_csv(r'Generate_dta\N_test_consumption1.csv')
t_consumption = pd.read_csv(r'Generate_dta\t_consumption.csv')
N_rong_tag_test = pd.read_csv(r'Generate_dta\N_rong_tag_test.csv')

test = merge(S_test_user_info,N_test_user_info,how="left", left_on='user_id', right_on='user_id')
test = merge(test,relation1_test,how="left", left_on='user_id', right_on='user_id')
test = merge(test,relation2_test,how="left", left_on='user_id', right_on='user_id')
test = merge(test,N_test_consumption1,how="left", left_on='user_id', right_on='user_id')
test = merge(test,t_consumption,how="left", left_on='user_id', right_on='user_id')

Mat_Test = array(test.drop(['user_id'],axis=1).fillna(-1))

dtest = xgb.DMatrix(Mat_Test)

auc_calculate(Mat_Train,Mat_Labels,dtest)









