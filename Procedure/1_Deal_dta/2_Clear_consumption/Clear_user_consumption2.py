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


def cal_newest_consumption(consumption_recode):
    consumption = consumption_recode.drop('bill_id', axis = 1)
    t = [x for x in consumption.groupby('user_id')]
    t_consumption = pd.DataFrame(columns = consumption.columns)

    count = 0
    for i in range(len(t)):
        uinfo = t[i][1]
        t_consumption.loc[i] = consumption.ix[uinfo.index[-1]]
    return t_consumption


###
# 数据读取
def Read_dta(f):
    N_dta = []
    N_dta_columns = f.readline().strip()
    N_dta_columns = N_dta_columns.split('\t')
    while True:
        line = f.readline().strip()
        if line:
            line = line.split('\t')
            N_dta.append(line)
        else:
            break
    N_dta = DataFrame(N_dta,columns=N_dta_columns)
    return N_dta

filename = open('dta\\Original_dta\\consumption_recode.txt','r')
consumption_recode = Read_dta(filename)
u_train = pd.read_csv('dta\\Original_dta\\train.txt')
u_test = pd.read_csv('dta\\Original_dta\\test\\test.txt')

t_consumption = cal_newest_consumption(consumption_recode)
t_consumption.to_csv('dta\\Generate_dta\\t_consumption.csv',index=False)

end = time.clock()
print ('Running time: %s Seconds'%(end-start))





