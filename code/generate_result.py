#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
#超算加下面这句话，平时不加
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import heapq
import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)
import sys
import numpy as np
import pandas as pd
import pickle
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing import sequence
from keras import backend as K
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau


sys.path.append('models')
from CNN import model_conv1D_, ABCNN2, dssm, fuck_model_conv1D_
from RNN import rnn_v1, Siamese_LSTM, my_rnn, fuck_my_rnn
from ESIM import esim, decomposable_attention, fuck_esim, fuck_decomposable_attention
from ABCNN import ABCNN
from bimpm import bimpm
from MatchZoo import *
sys.path.append('utils/')
sys.path.append('feature/')
import config
import fuck_config
from Feats import data_2id, add_hum_feats
from help import *  # score, train_batch_generator, train_batch_generator3,train_batch_generator5,train_test, get_X_Y_from_df
from CutWord import fuck_read_cut
from help_add import fuck_add

A = list(pd.read_csv('./data/test/row/standard_question.csv', encoding='gbk')['a'])

def get_model():

    lr = 0.01
    model_1 = fuck_decomposable_attention()
    #model_1.load_weights('./data/share/single/dp_feats_26_embed_cnn.h5')
    
    model_2 = fuck_my_rnn()
    #model_2.load_weights('./data/share/single/dp_feats_26_embed_esim.h5')


    return model_1, model_2, lr


def load_data():
    path = fuck_config.origin_csv
    print('load data')
    data = fuck_read_cut(path)  # cut word
    data = data_2id(data)  # 2id
    data = add_hum_feats(data, fuck_config.test_featdires)  # 生成特征并加入
    return data


def train_model(x_data, y_data, model_1, model_2, lr):#, bst_model_path):
    # print('x_train.shape',x_train[0])
    # print('y_train.shape',np.array(y_train).shape)
    # model_checkpoint = ModelCheckpoint(
    #     bst_model_path, monitor='val_F1', save_best_only=True, save_weights_only=True, mode='max')
    # early_stopping = EarlyStopping(monitor='val_F1', patience=4,
    #                                mode='max')
    # change_lr = ReduceLROnPlateau(
    #     monitor='val_F1', mode='max', factor=0.1, epsilon=0.001, min_lr=0.0001, patience=1)

    

    merge_input=[]
    merge_input.extend(model_1.input)
    merge_input.extend(model_2.input)

    out1=model_1.output
    out2=model_2.output
    x=concatenate([out1,out2],axis=1)
    Merge_model=Model(merge_input,output=x)


    inp=Merge_model.input
    x=Merge_model.output
    dense = BatchNormalization()(x)
    dense = Dropout(0.5)(dense)
    result=Dense(1,activation='sigmoid')(dense)
    model=Model(input=inp,output=result)

    adam=Adam(lr=0.0005,beta_1=0.95,beta_2=0.999,epsilon=1e-8)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=[Precision, Recall, F1])
    model.load_weights('./data/share/stack/word_dp_feats_26_rnn0_dam_1.h5')# 42, 3, 0, 187

    K.set_value(model.optimizer.lr, lr)

    x_data = x_data + x_data

    # model.fit(x_train, y_train,
    #           epochs=10,
    #           validation_data=(x_dev, y_dev),
    #           batch_size=config.batch_size,
    #           class_weight={0: 1, 1: 3},
    #           #callbacks=[model_checkpoint, early_stopping, change_lr,
    #                      # TensorBoard(log_dir='data/log_dir'),
    #                      #],
    #           )
    #x_dev = np.expand_dims(np.array(x_dev), axis=0)
    preds = model.predict(x_data)

    #print('row_pred:')
    #print(preds[:20])
    print('=========================')
    return preds, y_data

#此函数用于降维，废弃了
def my_squeeze(a):
    #把列表转为字符串
    b = str(a)
    #替换掉'['和']'
    b = b.replace('[','')
    b = b.replace(']','')
    #最后转化成列表
    a = list(eval(b))
    return a

def top_5(myList, fuck_add_pred):
    tempp = [i[0] for i in myList]
    temppp = [ tempp[j]+fuck_add_pred[j]  for j in range(len(tempp))]
    re1 = list(map(temppp.index, heapq.nlargest(5, temppp))) #求最大的五个索引
    return re1

def get_half(l):
    n_l = len(l) // 2
    return l[:n_l]


def do_single_train(model_name, model_1, model_2, lr):
    print('model name', model_name)
    # bst_model_path = config.model_dir + \
    #     "dp_feats_%d_embed_%s.h5" % (len(config.feats), model_name)
    data = load_data()
    #train, dev = train_test(data)#划分训练集和验证集
    x_data, y_data = get_X_Y_from_df(
        data, config.data_augment, False)#config.shuffer)#data_augment决定是否3还是5个Input（5个对应esim和cnn，3个对应其他）
    #x_dev, y_dev = get_X_Y_from_df(dev, config.data_augment, False)#False, False)

    preds, y_data = train_model(x_data, y_data, model_1, model_2, lr)#, bst_model_path)

    P = []
    top5 = []
    Preds = []

    n_a = len(A)

    n_q = len(preds) // n_a  #n_q: 1320,两倍
    print('=====================')
    print('=====================')
    print('n_q:', n_q)
    print('=====================')
    print('=====================')

    Q = list(pd.read_csv('./data/test/row/test.csv', encoding='gbk')['q'])
    Q = Q+Q

    for i in range(n_q):
        temp = preds[i*n_a : (i+1)*n_a]
        sent = Q[i]
        fuck_add_pred = fuck_add(sent, n_a)
        #貌似我的降维方法会出错
        #temp = my_squeeze(temp)
        #改用这种获得最大值的索引
        #print('temp[:10]', temp[:10])
        #print('\n')
        #pred_a = np.argmax(temp)

        c = [temp[w][0]+fuck_add_pred[w] for w in range(0,len(temp))]

        p_index = np.argmax( c )
        P.append( p_index )
        top5.append( top_5(temp, fuck_add_pred) )
        Preds.append( temp[p_index][0] )

    P = get_half(P)
    top5 = get_half(top5)
    Preds = get_half(Preds)

    print('P:', P)
    print('top5:', top5)
    print('Preds:', Preds)
    return P, top5, Preds

# def do_train_cv(model_name, model, kfolds, lr):
#     data = load_data()
#     X_train, Y_train = get_X_Y_from_df(
#         data, config.data_augment, config.shuffer)
#     make_train_cv_data(X_train, Y_train, model_1, model_2, model_name, kfolds, lr)



    
    


def fuck_P(P):
    fuck = []
    for p in P:
        fuck.append(A[p])
    return fuck

def fuck_top5(top5):
    fuck_5 = []
    for i in top5:
        fuck_temp = []
        for j in i:
            fuck_temp.append(A[j])
        fuck_5.append(fuck_temp)
    return fuck_5


def single_train(model_name):
    model_1, model_2, lr = get_model()
    P, top5, Preds = do_single_train(model_name, model_1, model_2, lr)
    Q = list(pd.read_csv('./data/test/row/test.csv', encoding='gbk')['q'])
    temp = pd.DataFrame(columns=['q', 'a', 'top5'],index=None)
    temp['q'] = Q
    temp['a'] = fuck_P(P)
    temp['top5'] = fuck_top5(top5)
    temp['top1_Preds'] =Preds
    temp.to_csv('./result.csv', encoding='gbk', index=None)

if __name__ == '__main__':

    model_name = 'rnn0_dam'
    single_train(model_name)
