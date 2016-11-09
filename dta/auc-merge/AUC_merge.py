#!/usr/bin/python
# coding=utf-8

import os
import numpy as np
from numpy import *
from pandas import *
import pandas as pd
import csv
from sklearn import metrics
import math


os.chdir(r'E:\PycharmProjects\Rong360\dta\Submit_forecast_dta\merge')


def rank_transform(data):
    for i in range(len(data)):
        data[i] = math.log(data[i],1.1)
    data = math.log(int(len(data)/2*2.3),1.1) - data
    return DataFrame(data)

def semple_pred(data):
    Pred_rank = pd.DataFrame(data.user_id,columns=['user_id'])
    N_rank_dta = DataFrame()
    N_rank_dta = data['probability'].rank(method='max')
    
    Pred_rank['probability'] = data.probability
    Pred_rank['r_'+'probability'] = rank_transform(N_rank_dta)
    Pred_rank['probability2'] = 2*(Pred_rank['probability']/Pred_rank['r_probability'].astype(float))
    print (max(Pred_rank['probability2']))
    print (min(Pred_rank['probability2']))
    
    return Pred_rank



Pred_1 = read_csv('P_145.csv')
Pred_2 = read_csv('P_1007.csv')

Pred_rank_1 = semple_pred(Pred_1)
Pred_rank_2 = semple_pred(Pred_2)
Pred_rank = DataFrame()
Pred_rank['user_id'] = Pred_rank_1['user_id']
Pred_rank['probability'] = 0.5*Pred_rank_1['probability2'] + 0.5*Pred_rank_2['probability2']

print (max(Pred_rank['probability']))
print (min(Pred_rank['probability']))

Pred_rank.to_csv(u'rank_1008.csv',index=False)



