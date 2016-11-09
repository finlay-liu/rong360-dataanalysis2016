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



'''
params=[max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective="reg:linear",
                 nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None)]

'''
def parameter_optimize(dtrain ,dval ,dtest ,target_test):

    param = {
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
	'seed':520341,
	'nthread':4
        }
    watchlist  = [(dtrain,'train'),(dval,'val')]
    bst = xgb.train(param, dtrain, num_boost_round = 46 ,evals=watchlist,\
                    verbose_eval=1)
    
    pred = bst.predict(dtest)
    auc_value = metrics.roc_auc_score(target_test, pred)
    return auc_value,bst

def auc_calculate(Mat_Train,Mat_Labels):
    kf = KFold(len(Mat_Train), n_folds=10)
    gather_auc = []
    i = 1
    for train_index, test_index in kf:
        print (i)
        i = i+1
        X_train, X_test = Mat_Train[train_index], Mat_Train[test_index]
        y_train, y_test = Mat_Labels[train_index], Mat_Labels[test_index]
        dtrain = xgb.DMatrix(X_train, y_train)
        dval = xgb.DMatrix(X_test,y_test)
        dtest = xgb.DMatrix(X_test)
        auc_value,bst = parameter_optimize(dtrain, dval ,dtest ,y_test)
        gather_auc.append(auc_value)
        print ('\n')
    mean_auc = mean(gather_auc)
    return mean_auc,bst


os.chdir(r'E:\PycharmProjects\Rong360\dta')

# 训练集

train = pd.read_csv(r'Generate_dta\N_rong_tag_train.csv')
train = train.fillna(-1)
Mat_Train = train.drop(['user_id','lable'],axis=1)
Mat_Train = array(Mat_Train)
Mat_Labels = train['lable'].astype(int)


mean_auc,bst = auc_calculate(Mat_Train,Mat_Labels)
print "恭喜您的AUC值已然达到:",mean_auc


























