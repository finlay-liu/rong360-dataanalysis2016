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





# 训练集
S_train_user_info = pd.read_csv(r'Generate_dta\S_train_user_info.csv')
N_train_user_info = pd.read_csv(r'Generate_dta\N_train_user_info.csv').drop(['lable'],axis=1)
relation1_train = pd.read_csv(r'Generate_dta\0909relation1_train.csv')
relation2_train = pd.read_csv(r'Generate_dta\0909relation2_train.csv')
N_train_consumption1 = pd.read_csv(r'Generate_dta\N_train_consumption1.csv').drop(['lable'],axis=1)
t_consumption = pd.read_csv(r'Generate_dta\t_consumption.csv')
'''
rong_tag_train = pd.read_csv(r'Generate_dta\N_rong_tag_train.csv').drop(['lable'],axis=1)
N_rong_tag_train_var = pd.read_excel(r'Stat_importance_var.xls')
N_rong_tag_train_var = N_rong_tag_train_var[N_rong_tag_train_var['Importance']>1]
N_rong_tag_train = rong_tag_train.reindex(columns = N_rong_tag_train_var['Feature'].values)
N_rong_tag_train['user_id'] = rong_tag_train['user_id']
N_rong_tag_train = N_rong_tag_train.replace([None], [-1])
'''
train = merge(S_train_user_info,N_train_user_info,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation1_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation2_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,N_train_consumption1,how="left", left_on='user_id', right_on='user_id')
train = merge(train,t_consumption,how="left", left_on='user_id', right_on='user_id')

train = train.replace([None], [-1])
train['category_null'] = (train<0).sum(axis=1)
train = train[train['category_null'] < 187]
train = DataFrame(train.values,columns=train.columns)
#train = merge(train,N_rong_tag_train,how="left", left_on='user_id', right_on='user_id')

Mat_Train = train.drop(['user_id','lable','category_null'],axis=1)




# 测试集数据
    # 测试集
S_test_user_info = pd.read_csv(r'Generate_dta\S_test_user_info.csv')
N_test_user_info = pd.read_csv(r'Generate_dta\N_test_user_info.csv')
relation1_test = pd.read_csv(r'Generate_dta\0909relation1_test.csv')
relation2_test = pd.read_csv(r'Generate_dta\0909relation2_test.csv')
N_test_consumption1 = pd.read_csv(r'Generate_dta\N_test_consumption1.csv')
t_consumption = pd.read_csv(r'Generate_dta\t_consumption.csv')
'''
rong_tag_test = pd.read_csv(r'Generate_dta\N_rong_tag_test.csv')
N_rong_tag_train_var = pd.read_excel(r'Stat_importance_var.xls')
N_rong_tag_train_var = N_rong_tag_train_var[N_rong_tag_train_var['Importance']>10]
N_rong_tag_test = rong_tag_test.reindex(columns = N_rong_tag_train_var['Feature'].values)
N_rong_tag_test['user_id'] = rong_tag_test['user_id']
N_rong_tag_test = N_rong_tag_test.replace([None], [-1])
'''
test = merge(S_test_user_info,N_test_user_info,how="left", left_on='user_id', right_on='user_id')
test = merge(test,relation1_test,how="left", left_on='user_id', right_on='user_id')
test = merge(test,relation2_test,how="left", left_on='user_id', right_on='user_id')
test = merge(test,N_test_consumption1,how="left", left_on='user_id', right_on='user_id')
test = merge(test,t_consumption,how="left", left_on='user_id', right_on='user_id')
#test = merge(test,N_rong_tag_test,how="left", left_on='user_id', right_on='user_id')

Mat_Test = array(test.drop(['user_id'],axis=1).fillna(-1))



'''
for i in range(150,197):
    print i
    print len(train[train['category_null']>i])
    print "\n"

for i in range(1,len(train.columns)):
    print train[train.columns[i]].value_counts()
    print '\n'
'''









