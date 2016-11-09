# coding=utf-8

'''
author: ShiLei Miao
analyses and build model about Rong360
'''

import numpy as np
from numpy import *
import pandas as pd
from pandas import *
import os
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from pylab import rcParams
import operator
from matplotlib import pylab as plt

os.chdir('E:\PycharmProjects\Rong360')


# 训练集数据
train_xy = pd.read_csv(r'dta\Generate_dta\N_rong_tag_train.csv')
train_xy = train_xy.fillna(-1)

Mat_Train = train_xy.drop(['user_id','lable'],axis=1)
Mat_Labels = train_xy['lable']
N_train, N_test, target_train, target_test = train_test_split(\
    Mat_Train, Mat_Labels, test_size=0.3)

dtrain_xy = xgb.DMatrix(N_train, target_train)
dtrain_x = xgb.DMatrix(N_train)
dtest = xgb.DMatrix(N_test)


# Xgb训练
num_round = 46
params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'gamma':15,
    'eta': 0.045,
    'lambda':17,
    'subsample':0.89,
    'colsample_bytree':0.89,
    'max_depth':6,
    'scale_pos_weight': 1.2,

    'missing':-1,
    'seed':520,
    'nthread':4
    }

bst = xgb.train(params, dtrain_xy, num_round)
bst.save_model(r'Procedure\2_M1\rongtag-select-feature\xgb.model')
bst.dump_model(r'Procedure\2_M1\rongtag-select-feature\model_structure.txt')
pred_train = bst.predict(dtrain_x)
pred_test = bst.predict(dtest)

auc_train = metrics.roc_auc_score(target_train, pred_train)
auc_test = metrics.roc_auc_score(target_test, pred_test)


print auc_train
print auc_test

# xgb特征
importances = bst.get_fscore()
importance_frame = pd.DataFrame\
                   ({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)


feature_info = {}
features = importance_frame.Feature.values
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


var_feature_info1.to_excel(r'Procedure\2_M1\rongtag-select-feature\Stat_importance_var.xls',encoding='GBK')


print var_feature_info1.shape

