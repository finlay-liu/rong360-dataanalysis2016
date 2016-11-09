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
import operator
from matplotlib import pylab as plt
from pylab import rcParams
from sklearn.cross_validation import train_test_split

os.chdir(r'E:\PycharmProjects\Rong360')


# 训练集
S_train_user_info = pd.read_csv(r'dta\Generate_dta\S_train_user_info.csv')
N_train_user_info = pd.read_csv(r'dta\Generate_dta\N_train_user_info.csv').drop(['lable'],axis=1)
relation1_train = pd.read_csv(r'dta\Generate_dta\0909relation1_train.csv')
relation2_train = pd.read_csv(r'dta\Generate_dta\0909relation2_train.csv')
N_train_consumption1 = pd.read_csv(r'dta\Generate_dta\N_train_consumption1.csv').drop(['lable'],axis=1)
t_consumption = pd.read_csv(r'dta\Generate_dta\t_consumption.csv')

train = merge(S_train_user_info,N_train_user_info,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation1_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation2_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,N_train_consumption1,how="left", left_on='user_id', right_on='user_id')
train = merge(train,t_consumption,how="left", left_on='user_id', right_on='user_id')

train = train.replace([None], [-1])
train['category_null'] = (train<0).sum(axis=1)
train_xy = train[train['category_null'] < 187]

# 训练集数据
Mat_Train = train_xy.drop(['user_id','lable','category_null'],axis=1)
Mat_Labels = train_xy['lable']
dtrain = xgb.DMatrix(Mat_Train, Mat_Labels)


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
bst.save_model(r'Procedure\2_M1\2-train_clear_again\xgb_1003.model')
bst.dump_model(r'Procedure\2_M1\2-train_clear_again\model_structure.txt')

# xgb特征
importances = bst.get_fscore()
importance_frame = pd.DataFrame\
                   ({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)


feature_info = {}
features = importance_frame.Feature.values
#features.remove('user_id')
for feature in features:
    n_max = train_xy[feature].max()
    n_min = train_xy[feature].min()
    n_mean = train_xy[feature].mean()
    n_std = train_xy[feature].std()
    n_null = len(train_xy[train_xy[feature]<0])  #number of null
    n_quantile_25 = np.percentile(train_xy[feature],25)
    n_quantile_50 = np.percentile(train_xy[feature],50)
    n_quantile_75 = np.percentile(train_xy[feature],75)
    feature_info[feature] = [n_min,n_max,n_mean,n_std,n_null,n_quantile_25,n_quantile_50,n_quantile_75]


N_feature = ['n_min','n_max','n_mean','n_std','n_null','n_quantile_25','n_quantile_50','n_quantile_75']
var_feature_info = DataFrame(feature_info,index=N_feature).T
var_feature_info['Feature'] = var_feature_info.index
var_feature_info = DataFrame(var_feature_info.values,columns=var_feature_info.columns)
var_feature_info1 = merge(importance_frame,var_feature_info,how='left',left_on='Feature',right_on='Feature')

var_feature_info1.to_excel(r'Procedure\2_M1\2-train_clear_again\model_var_importance.xls',encoding='GBK')



