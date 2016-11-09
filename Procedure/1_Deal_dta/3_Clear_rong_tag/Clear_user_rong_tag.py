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
rong_tag = pd.read_csv('dta\\Original_dta\\rong_tag.txt')
train = pd.read_csv('dta\\Original_dta\\train.txt')
test = pd.read_csv('dta\\Original_dta\\test\\test.txt')

rong_tag['rong_tag'] = (rong_tag['rong_tag'] - 300000)/7
rong_tag = rong_tag[rong_tag['rong_tag']<300]




def Dummy(data, variable):
    One_hot_dta = get_dummies(data, prefix=variable)
    return One_hot_dta

def Dummy_Master(data):
    names = data.columns
    for i in names:
        if i != "user_id" and len(data[i].value_counts()) > 2:
            One_hot_dtas = Dummy(data[i], i)
            data = data.join(One_hot_dtas)
            data = data.drop(i, axis=1)
    return data

# 将类别变量进行one_hot编码
rong_tag = Dummy_Master(rong_tag)

N_rong_tag = DataFrame()
for i in range(1,rong_tag.shape[1]):
    Var_N = rong_tag.columns[i]
    N_rong_tag[Var_N] = rong_tag.groupby('user_id')[Var_N].max()
N_rong_tag['user_id'] = N_rong_tag.index

rong_tag_train = pd.merge(train,N_rong_tag, how="left", left_on='user_id', right_on='user_id')
rong_tag_test = pd.merge(test,N_rong_tag, how="left", left_on='user_id', right_on='user_id')


rong_tag_train.to_csv('dta\\Generate_dta\\N_rong_tag_train.csv',index=False)
rong_tag_test.to_csv('dta\\Generate_dta\\N_rong_tag_test.csv',index=False)








'''
rong_tag_train = pd.merge(rong_tag, train, how="inner", left_on='user_id', right_on='user_id')
rong_tag_test = pd.merge(rong_tag, test, how="inner", left_on='user_id', right_on='user_id')

tag_train = rong_tag_train['rong_tag'].unique()
tag_test = rong_tag_test['rong_tag'].unique()

# 共同的tag
commmon_tag = list(set(tag_train) & set(tag_test))



dta = DataFrame(rong_tag['rong_tag'].value_counts())
dta['index'] = dta.index
dta['rong_tag_count'] = dta['rong_tag']
dta = DataFrame(dta.values,columns=dta.columns).drop(['rong_tag'],axis=1)
dta2 = dta[dta['rong_tag_count']>200]

com_tag = []
for i in commmon_tag:
    if i in dta2['index'].values:
        com_tag.append(i)



rong_tag_train2 = rong_tag_train.loc[rong_tag_train['rong_tag'].isin(com_tag)]
rong_tag_test2 = rong_tag_test.loc[rong_tag_test['rong_tag'].isin(com_tag)]



def Summarizing_basic_information(N_data,variable):

    N_data[variable] = N_data[variable].astype(float)
    data = DataFrame()
    data['min_'+variable] = N_data.groupby('user_id')[variable].min()
    data['max_'+variable] = N_data.groupby('user_id')[variable].max()
    data['sum_'+variable] = N_data.groupby('user_id')[variable].sum()
    data['mean_'+variable] = N_data.groupby('user_id')[variable].mean()
    data['std_'+variable] = N_data.groupby('user_id')[variable].std()
    data['var_'+variable] = N_data.groupby('user_id')[variable].var()
    data['count_'+variable] = N_data.groupby('user_id')[variable].count()
    data['user_id'] = data.index
    data = DataFrame(data.values,columns=data.columns)
    data = data.reindex(columns=['user_id','min_'+variable,'max_'+variable,'sum_'+variable,'mean_'+variable,'std_'+variable,\
                                 'var_'+variable,'count_'+variable])

    return data


## 将数据集进行区间估计
def C_confidence_interval(dta,variable):
    dta[variable+u'_upper'] = 0
    dta[variable+u'_lower'] = 0
    t_table = pd.read_excel(u'dta\\Original_dta\\user_dta\\t检验临界值表.xls','t_table')
    z_a = 1.96
    for i in range(len(dta)):
        if dta.ix[i][u'count_'+variable] < 30:
            t_a = t_table.ix[dta.ix[i][u'count_'+variable]-1]['a3']
            dta.loc[i,variable+u'_upper'] = dta.ix[i]['mean_'+variable] + \
                                            t_a*dta.ix[i]['std_'+variable]/math.sqrt(dta.ix[i][u'count_'+variable])
            dta.loc[i,variable+u'_lower'] = dta.ix[i]['mean_'+variable] - \
                                            t_a*dta.ix[i]['std_'+variable]/math.sqrt(dta.ix[i][u'count_'+variable])
        if dta.ix[i][u'count_'+variable] >= 30:
            dta.loc[i,variable+u'_upper'] = dta.ix[i]['mean_'+variable] + \
                                            z_a*dta.ix[i]['std_'+variable]/math.sqrt(dta.ix[i][u'count_'+variable])
            dta.loc[i,variable+u'_lower'] = dta.ix[i]['mean_'+variable] - \
                                            z_a*dta.ix[i]['std_'+variable]/math.sqrt(dta.ix[i][u'count_'+variable])
    return dta
'''

'''
N_rong_tag_train2 = Summarizing_basic_information(rong_tag_train2,'rong_tag')
N_rong_tag_train = C_confidence_interval(N_rong_tag_train2,'rong_tag')

N_rong_tag_test2 = Summarizing_basic_information(rong_tag_test2,'rong_tag')
N_rong_tag_test = C_confidence_interval(N_rong_tag_test2,'rong_tag')


tag_train = pd.merge(train, N_rong_tag_train, how="left", left_on='user_id', right_on='user_id')
tag_test = pd.merge(test, N_rong_tag_test, how="left", left_on='user_id', right_on='user_id')
'''


#tag_train.to_csv('dta\\Generate_dta\\N_rong_tag_train.csv',index=False)
#tag_test.to_csv('dta\\Generate_dta\\N_rong_tag_test.csv',index=False)


end = time.clock()
print 'Running time: %s Seconds'%(end-start)


















