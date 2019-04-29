import pandas as pd
import numpy as np

A = list(pd.read_csv('./data/test/row/standard_question.csv', encoding='gbk')['a'])
Q = list(pd.read_csv('./data/test/row/test.csv', encoding='gbk')['q'])

data = pd.DataFrame(columns=['q', 'a'],index=None)
# data['q'] = ['q1', 'q2']
# data['a'] = ['a1', 'a2']
# data.head()
for q in Q:
    temp = pd.DataFrame(columns=['q', 'a'],index=None)
    temp['q'] = [q for i in range(209)]
    temp['a'] = A
    temp['TARGET'] = 0
    #print(temp)
    #break
    data = pd.concat([data, temp], axis=0, ignore_index=True)

data.to_csv('./data/test/row/test_concat.csv', encoding='gbk', index=None)