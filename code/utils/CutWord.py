#/usr/bin/env python
# coding=utf-8
import os
import sys
import numpy as np
import pandas as pd
import jieba
import re 

sys.path.append('utils/')
import config
from pinyin import PinYin
str2pinyin = PinYin()
jieba.load_userdict('data/jieba/jieba_dict.txt')
stopwords = [line.strip() for line in open('data/jieba/stops.txt', 'r', encoding='utf-8').readlines()]
# stopwords = [w.decode('utf8') for w in stopwords]
# stopwords=[]
#if config.cut_char_level:
stopwords = [u'？', u'。', u'，',]

use_pinyin =False


def clean_str(x):
    punc = "蚂蚁  了 吗  的 ！？。，：；. 花呗"
    return re.sub('|'.join(punc.strip().split()), "", x)
    
    #data = re.sub(r"[(u'蚂蚁')(u'了')(u'吗')(u'的')(u'！')(u'？')(u'。')(u'，')(u'：')(u'；')(u'.')]", "", x)
    #return data
    #return x


def cut_single(x,cut_char_level):
    x = clean_str(x)
    res = []
    if cut_char_level:
        setence_seged = list(x.strip())
        #print(setence_seged)
    else:
        setence_seged = jieba.cut(x.strip(), HMM=True)
        #setence_seged=jieba.cut_for_search(x.strip(),HMM=Truec)

        #import jieba.analyse
        #setence_seged = jieba.analyse.extract_tags(x.strip(),topK=5,withWeight=False,allowPOS=['n','v'])
    
    for word in setence_seged:
        if word not in stopwords:
            my_word = word
            if use_pinyin:
                my_word = str2pinyin.char2pinyin(my_word)
            res.append(my_word)
    return res
def more(data,n):
    pass
def cut_word(path):

    data = pd.read_csv('./data/row/data_shuffle.csv', encoding='gbk')
    #data['id'] = range(len(data))#重塑id
    data['TARGET'] = data['TARGET'].fillna(0)
    data['q_cut'] = data['q'].map(lambda x: cut_single(x,cut_char_level=True))
    data['a_cut'] = data['a'].map(lambda x: cut_single(x,cut_char_level=True))

    data['q_cut_word'] = data['q'].map(lambda x: cut_single(x,cut_char_level=False))
    data['a_cut_word'] = data['a'].map(lambda x: cut_single(x,cut_char_level=False))
    print('cut done')
    #data.to_csv('atec_nlp_sim_train_3.csv', encoding='utf-8')
    print(data.shape)
    return data

def fuck_cut_word(path):

    data = pd.read_csv('./data/test/row/test_concat.csv', encoding='gbk')
    #data['id'] = range(len(data))#重塑id
    data['TARGET'] = 0
    data['q_cut'] = data['q'].map(lambda x: cut_single(x,cut_char_level=True))
    data['a_cut'] = data['a'].map(lambda x: cut_single(x,cut_char_level=True))

    data['q_cut_word'] = data['q'].map(lambda x: cut_single(x,cut_char_level=False))
    data['a_cut_word'] = data['a'].map(lambda x: cut_single(x,cut_char_level=False))
    print('cut done')
    #data.to_csv('atec_nlp_sim_train_3.csv', encoding='utf-8')
    print(data.shape)
    return data
    
    
def read_cut(path):
    if not os.path.exists('./data/cache/train_cut.csv'):
        data = cut_word(path)
        data.to_csv('./data/cache/train_cut.csv', encoding='gbk', index=None)
    # pwd = os.getcwd()
    # os.chdir(os.path.dirname(path))
    # data = pd.read_csv(os.path.basename(path),encoding='gbk')
    # os.chdir(pwd)
    data = pd.read_csv('./data/cache/train_cut.csv', encoding='gbk')
    return data

def fuck_read_cut(path):
    if not os.path.exists('./data/test/test_cut.csv'):
        data = fuck_cut_word(path)
        data.to_csv('./data/test/test_cut.csv', encoding='gbk', index=None)
    # pwd = os.getcwd()
    # os.chdir(os.path.dirname(path))
    # data = pd.read_csv(os.path.basename(path),encoding='gbk')
    # os.chdir(pwd)
    data = pd.read_csv('./data/test/test_cut.csv', encoding='gbk')
    return data

if __name__ == '__main__':
    path = '../' + config.origin_csv
    #read_data(path)
    cut_word(path)
