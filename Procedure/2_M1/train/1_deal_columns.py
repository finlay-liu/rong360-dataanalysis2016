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



def calCorr2(df):
    col = []
    for i in df.columns:
        if i == 'user_id' or i == 'lable':
            continue
        for j in df.columns:
            if df[i].dtype  == np.object or df[j].dtype  == np.object:
                continue
            # if i != j and len(df[i].dropna()) == len(df[j].dropna()) and sum(pd.isnull(df[i]) == pd.isnull(df[j])) == len(df):
            if i != j and len(df[i].dropna()) == len(df[j].dropna()):
                corr = np.corrcoef(df[i].dropna(), df[j].dropna())[0, 1]
                if corr >= 1.0:
                    print '{0}, {1}, \t{2}'.format(i, j, corr)
                    col.append(i)
        print "\n"
    return list((set(col)))




def calCorr(df):
    N_columns = df.columns
    col = []
    for i in range(len(N_columns)):
        if N_columns[i] == 'user_id' or N_columns[i] == 'lable' or (N_columns[i] in col):
            continue
        for j in range(i+1,len(N_columns)):
            corr = np.corrcoef(df[N_columns[i]], df[N_columns[j]])[0, 1]
            if corr >= 1.0:
                print '{0}, {1}, \t{2}'.format(N_columns[i], N_columns[j], corr)
                col.append(N_columns[j])
        #print "\n"
    return list((set(col)))







os.chdir(r'E:\PycharmProjects\Rong360\dta')


# 训练集
S_train_user_info = pd.read_csv(r'Generate_dta\S_train_user_info.csv')
N_train_user_info = pd.read_csv(r'Generate_dta\N_train_user_info.csv')
relation1_train = pd.read_csv(r'Generate_dta\0909relation1_train.csv')
relation2_train = pd.read_csv(r'Generate_dta\0909relation2_train.csv')
N_train_consumption1 = pd.read_csv(r'Generate_dta\N_train_consumption1.csv')
t_consumption = pd.read_csv(r'Generate_dta\t_consumption.csv')
N_rong_tag_train = pd.read_csv(r'Generate_dta\N_rong_tag_train2.csv')

N_train_user_info = N_train_user_info.drop(['lable'],axis=1)
N_train_consumption1 = N_train_consumption1.drop(['lable'],axis=1)
N_rong_tag_train = N_rong_tag_train.drop(['lable'],axis=1)

train = merge(S_train_user_info,N_train_user_info,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation1_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,relation2_train,how="left", left_on='user_id', right_on='user_id')
train = merge(train,N_train_consumption1,how="left", left_on='user_id', right_on='user_id')
train = merge(train,t_consumption,how="left", left_on='user_id', right_on='user_id')
train = merge(train,N_rong_tag_train,how="left", left_on='user_id', right_on='user_id')

train = train.fillna(-1)

deal_columns = calCorr(train)
DataFrame(deal_columns,columns=['var']).to_csv('N_deal_columns.csv',index=False)










