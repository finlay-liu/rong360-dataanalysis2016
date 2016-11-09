import os
import csv
import math
import time
import igraph
import numpy as np
import pandas as pd

import os
import math
import numpy as np
import pandas as pd
import time

relation1 = pd.read_csv('../data/relation1.txt')
relation2 = pd.read_csv('../data/relation2.txt')

train = pd.read_csv('../data/train.txt')
test = pd.read_csv('../data/test.txt')


def genRealtionAvl(relation, train, test):
    relation_user = set(relation['user1_id'].unique()) | set(relation['user2_id'].unique())
    
    re2_train = set()
    for index, row in train.iterrows():
        if row['user_id'] not in relation_user:
            re2_train.add(index)
    
    re1_test = set()
    for index, row in test.iterrows():
        if row['user_id'] not in relation_user:
            re1_test.add(index)
    
    return train.drop(re2_train), test.drop(re1_test) 

re2_train, re2_test = genRealtionAvl(relation2, train, test)

re2_map = igraph.Graph.TupleList(csv.reader(open("../data/relation2.txt")))

# set lable
for index, row in re2_train.iterrows():
    re2_map.vs[re2_map.vs.find(row['user_id']).index]['lable'] = row['lable']

train_vertex = [re2_map.vs.find(x).index for x in re2_train['user_id'].tolist()]

def searchRelation1(user_id, depth):
    if depth == 1:
        return [x for x in re2_map.neighbors(user_id)]
    else:
        neighbors = []
        for neighbor in searchRelation1(user_id, 1):
            neighbors.extend(list(searchRelation1(neighbor, depth - 1)))
        return set(neighbors)

def calRelation1(cal_set):
    res = []
    tlen = len(cal_set)
    for index, row in cal_set.iterrows():
        friends = searchRelation1(re2_map.vs.find(row['user_id']).index, 1)
        friends2 = searchRelation1(re2_map.vs.find(row['user_id']).index, 2)
        
        t = [len(friends), len(friends2)]
        t = [str(x) for x in t]
        res.append(row['user_id'] + ',' + ','.join(t))
        # if len(res) % 100 == 0:
        print len(res), tlen
        print res[-1]
    return res

relation2_train = calRelation1(re2_train)
relation2_test = calRelation1(re2_test)

import codecs
def writeResultFile(content, path):
    fp = codecs.open('./data/' + path, 'w', 'utf-8')
    fp.write('user_id,re2_1u_friends,re2_2u_friends\n')
    for i in content:
        fp.write('%s\n' % i)
    fp.close()

print 'Writing File...'
writeResultFile(relation2_train, 'relation2_train.csv')
writeResultFile(relation2_test, 'relation2_test.csv')