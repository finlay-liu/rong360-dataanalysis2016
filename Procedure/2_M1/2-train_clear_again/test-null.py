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
train['sam_count'] = (train<0).sum(axis=1)



# 测试集数据
    # 测试集
S_test_user_info = pd.read_csv(r'dta\Generate_dta\S_test_user_info.csv')
N_test_user_info = pd.read_csv(r'dta\Generate_dta\N_test_user_info.csv')
relation1_test = pd.read_csv(r'dta\Generate_dta\0909relation1_test.csv')
relation2_test = pd.read_csv(r'dta\Generate_dta\0909relation2_test.csv')
N_test_consumption1 = pd.read_csv(r'dta\Generate_dta\N_test_consumption1.csv')
t_consumption = pd.read_csv(r'dta\Generate_dta\t_consumption.csv')

test = merge(S_test_user_info,N_test_user_info,how="left", left_on='user_id', right_on='user_id')
test = merge(test,relation1_test,how="left", left_on='user_id', right_on='user_id')
test = merge(test,relation2_test,how="left", left_on='user_id', right_on='user_id')
test = merge(test,N_test_consumption1,how="left", left_on='user_id', right_on='user_id')
test = merge(test,t_consumption,how="left", left_on='user_id', right_on='user_id')

test = test.replace([None], [-1])
test['sam_count'] = (test<0).sum(axis=1)

def null_stat(dta):
    null_stat = DataFrame(dta['sam_count'].value_counts())
    null_stat['null_count'] = null_stat.index
    null_stat = DataFrame(null_stat.sort_values(by='null_count',ascending=False).values,
                          columns=null_stat.columns)
    return null_stat


train_null_stat =  null_stat(train)
test_null_stat =  null_stat(test)

for i in range(10):
    xx = train_null_stat['null_count'][i]
    print xx
    print train[train['sam_count']==xx].lable.value_counts()
    print '\n'
'''
x = train[train['sam_count']==186].append(test[test['sam_count']==184])
x.to_csv(r'Procedure\2_M1\2-train_clear_again\question.csv',encoding='GBK')
'''

'''
dta = train[train['sam_count']==183]
for i in range(1,len(dta.columns)):
    print dta[dta.columns[i]].value_counts()
    print '\n'

'''


'''
for i in range(150,197):
    print i
    print len(train[train['category_null']>i])
    print "\n"

for i in range(1,len(train.columns)):
    print train[train.columns[i]].value_counts()
    print '\n'
'''









