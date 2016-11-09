#!/usr/bin/python
# coding=utf-8

import time
import numpy as np
from numpy import *
import pandas as pd
from pandas import *
import csv
import xgboost as xgb
import os


start =time.clock()
os.chdir(r'E:\PycharmProjects\Rong360\dta')



# 训练集
S_train_user_info = pd.read_csv(r'Generate_dta\S_train_user_info.csv')
N_train_user_info = pd.read_csv(r'Generate_dta\N_train_user_info.csv').drop(['lable'],axis=1)
relation1_train = pd.read_csv(r'Generate_dta\0909relation1_train.csv')
relation2_train = pd.read_csv(r'Generate_dta\0909relation2_train.csv')
N_train_consumption1 = pd.read_csv(r'Generate_dta\N_train_consumption1.csv').drop(['lable'],axis=1)
t_consumption = pd.read_csv(r'Generate_dta\t_consumption.csv')

train = merge(S_train_user_info,N_train_user_info,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation1_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation2_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,N_train_consumption1,how="left", left_on='user_id', right_on='user_id')
train = merge(train,t_consumption,how="left", left_on='user_id', right_on='user_id')

train = train.replace([None], [-1])
train['category_null'] = (train<0).sum(axis=1)
train = train[train['category_null'] < 187]
train = DataFrame(train.values,columns=train.columns)


Mat_Train = train.drop(['user_id','lable','category_null'],axis=1)
Mat_Train = array(Mat_Train)
Mat_Labels = DataFrame(train.lable).values[0:,0:]

labels = []
for item in Mat_Labels:
    labels.append(item[0])
labels = array(labels)



# 测试集数据
    # 测试集
S_test_user_info = pd.read_csv(r'Generate_dta\S_test_user_info.csv')
N_test_user_info = pd.read_csv(r'Generate_dta\N_test_user_info.csv')
relation1_test = pd.read_csv(r'Generate_dta\0909relation1_test.csv')
relation2_test = pd.read_csv(r'Generate_dta\0909relation2_test.csv')
N_test_consumption1 = pd.read_csv(r'Generate_dta\N_test_consumption1.csv')
t_consumption = pd.read_csv(r'Generate_dta\t_consumption.csv')

test = merge(S_test_user_info,N_test_user_info,how="left", left_on='user_id', right_on='user_id')
test = merge(test,relation1_test,how="left", left_on='user_id', right_on='user_id')
test = merge(test,relation2_test,how="left", left_on='user_id', right_on='user_id')
test = merge(test,N_test_consumption1,how="left", left_on='user_id', right_on='user_id')
test = merge(test,t_consumption,how="left", left_on='user_id', right_on='user_id')

Mat_Test = array(test.drop(['user_id'],axis=1).fillna(-1))

dtrain = xgb.DMatrix(Mat_Train, label=labels)
dtest = xgb.DMatrix(Mat_Test)
param={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'gamma':15,
    'eta': 0.045,
    'lambda':15,
    'subsample':0.886,
    'colsample_bytree':0.886,
    'max_depth':9,
    'scale_pos_weight': 1.2,

    'missing':-1,
    'seed':520341,
    'nthread':4
    }
num_round = 46
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

N_pred = DataFrame(preds,columns=['probability'])
N_pred['user_id'] = test['user_id']
N_pred = DataFrame(N_pred,columns=['user_id','probability'])

N_pred.to_csv(u'Submit_forecast_dta\\xgb_1002_1.csv',index=False)
end = time.clock()
print 'Running time: %s Seconds'%(end-start)






