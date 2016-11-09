# coding=utf-8

'''
author: ShiLei Miao
analyses and build model about NBA
'''

import numpy as np
from numpy import *
import pandas as pd
from pandas import *
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn import metrics


os.chdir(r'E:\PycharmProjects\Rong360\dta')

def loadDataSetT(path):
    data = pd.read_csv(path)
    dataSet = data.values[0:,2:]
    dataLabel = data.values[0:,1:2]        
    return dataSet,dataLabel

def transLabel(Mat_Labels):
    labels = []
    for item in Mat_Labels:
        labels.append(item[0])
    labels = array(labels)
    return labels



def P_YYYY(N_train, target_train, N_test, target_test):
    clf = RandomForestClassifier(n_estimators=300, random_state=520341, max_depth=9,\
                                 min_samples_split=3, class_weight='balanced_subsample')
    clf = clf.fit(N_train, target_train)

    pred = clf.predict_proba(N_test)
    pred = DataFrame(pred)[0].values
    N_auc = metrics.roc_auc_score(target_test, 1 - pred)
    print N_auc
    print '\n'
    return N_auc, clf

def preds_calculate(Mat_Train,Mat_Labels):
    kf = KFold(len(Mat_Train), n_folds=10)
    NN_auc = []
    for train_index, test_index in kf:
        X_train, X_test = Mat_Train[train_index], Mat_Train[test_index]
        y_train, y_test = Mat_Labels[train_index], Mat_Labels[test_index]
        N_auc, clf = P_YYYY(X_train, y_train,  X_test, y_test)
        NN_auc.append(N_auc)
    mean_auc = mean(NN_auc)
    print 'AUC均值：',mean_auc
    return mean_auc, clf



# 训练集
S_train_user_info = pd.read_csv(r'Generate_dta\S_train_user_info.csv')
N_train_user_info = pd.read_csv(r'Generate_dta\N_train_user_info.csv').drop(['lable'],axis=1)
relation1_train = pd.read_csv(r'Generate_dta\0909relation1_train.csv')
relation2_train = pd.read_csv(r'Generate_dta\0909relation2_train.csv')
N_train_consumption1 = pd.read_csv(r'Generate_dta\N_train_consumption1.csv').drop(['lable'],axis=1)
t_consumption = pd.read_csv(r'Generate_dta\t_consumption.csv')

#rong_tag 没有使用 【下面的数据是one-hot后的特征】
rong_tag_train = pd.read_csv(r'Generate_dta\N_rong_tag_train.csv').drop(['lable'],axis=1)
N_rong_tag_train_var = pd.read_excel(r'Stat_importance_var.xls')
N_rong_tag_train_var = N_rong_tag_train_var[N_rong_tag_train_var['Importance']>10]
N_rong_tag_train = rong_tag_train.reindex(columns = N_rong_tag_train_var['Feature'].values)
N_rong_tag_train['user_id'] = rong_tag_train['user_id']
N_rong_tag_train = N_rong_tag_train.replace([None], [-1])

train = merge(S_train_user_info,N_train_user_info,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation1_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation2_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,N_train_consumption1,how="left", left_on='user_id', right_on='user_id')
train = merge(train,t_consumption,how="left", left_on='user_id', right_on='user_id')


train = train.replace([None], [-1])
train['category_null'] = (train<0).sum(axis=1)

## 在统计的train跟test缺失的情况后，选择剔除用户的特征缺失个数为187的【基本都是product_id=2】
train = train[train['category_null'] < 187]
train = DataFrame(train.values,columns=train.columns)

train = merge(train,N_rong_tag_train,how="left", left_on='user_id', right_on='user_id')


Mat_Train = train.drop(['user_id','lable','category_null'],axis=1)
Mat_Train = array(Mat_Train)
Mat_Label = train['lable'].astype(int)

mean_auc, clf = preds_calculate(Mat_Train,Mat_Label)









