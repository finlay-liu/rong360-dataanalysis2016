# coding=utf-8

'''
author: ShiLei Miao
analyses and build model about NBA
'''

from numpy import *
import pandas as pd
from pandas import *
import os
import time

os.chdir(r'E:\PycharmProjects\Rong360')
start =time.clock()


###
# 数据读取
user_info = pd.read_csv('dta\\Original_dta\\user_info.txt')
u_train = pd.read_csv('dta\\Original_dta\\train.txt')
u_test = pd.read_csv('dta\\Original_dta\\test\\test.txt')


L_index = DataFrame(user_info.groupby('user_id')['tm_encode'].max())
L_index['index'] = L_index.index
f = lambda x:str(x)
L_index['qq'] = L_index['index'] + L_index.tm_encode.apply(f)
L_index = L_index.drop(['index','tm_encode'],axis=1)
L_index = DataFrame(L_index.values,columns=['qq'])


user_info['qq'] = user_info.user_id + user_info.tm_encode.apply(f)
user_info = user_info.drop_duplicates()

N_dta = merge(L_index,user_info,how='left',left_on='qq',right_on='qq')
N_dta = N_dta.drop(['qq'],axis=1)





N_user_info2 = DataFrame()
N_user_info2['count_user_info'] = user_info.groupby('user_id')['user_id'].count()
N_user_info2['user_id'] = N_user_info2.index 
N_user_info2 = DataFrame(N_user_info2.values,columns=N_user_info2.columns)
NN_dta = merge(N_user_info2,N_dta,how='left',left_on='user_id',right_on='user_id')
NN_dta = NN_dta.fillna(-1)


S_train_user_info = merge(u_train, NN_dta, how="left", left_on='user_id', right_on='user_id')
S_test_user_info = merge(u_test, NN_dta, how="left", left_on='user_id', right_on='user_id')

S_train_user_info.to_csv('dta\\Generate_dta\\S_train_user_info.csv',index=False)
S_test_user_info.to_csv('dta\\Generate_dta\\S_test_user_info.csv',index=False)





end = time.clock()
print 'Running time: %s Seconds'%(end-start)


















