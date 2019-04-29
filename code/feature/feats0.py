#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def magic1(train_in,cols=['q_cut_word','a_cut_word']):
    print('make feats0-------')
    # https://www.kaggle.com/jturkewitz/magic-features-0-03-gain
    train_in[cols[0]] = train_in[cols[0]].map(lambda x: ' '.join(x))
    train_in[cols[1]] = train_in[cols[1]].map(lambda x: ' '.join(x))

    train_orig = train_in.copy()

    df1 = train_orig[[cols[0]]].copy()
    df2 = train_orig[[cols[1]]].copy()

    df2.rename(columns={cols[1]: cols[0]}, inplace=True)

    train_questions = df1.append(df2)

    train_questions.reset_index(inplace=True, drop=True)

    questions_dict = pd.Series(
        train_questions.index.values, index=train_questions[cols[0]].values).to_dict()
    train_cp = train_orig.copy()

    comb = train_cp

    comb['q_hash'] = comb[cols[0]].map(questions_dict)
    comb['a_hash'] = comb[cols[0]].map(questions_dict)

    q1_vc = comb.q_hash.value_counts().to_dict()
    q2_vc = comb.a_hash.value_counts().to_dict()

    def try_apply_dict(x, dict_to_apply):
        try:
            return dict_to_apply[x]
        except KeyError:
            return 0
    # map to frequency space
    comb['q_freq'] = comb['q_hash'].map(
        lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))
    comb['a_freq'] = comb['a_hash'].map(
        lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))

    # Calculate derivative features

    comb['freq_mean'] = (comb['q_freq'] + comb['a_freq']) / 2
    comb['freq_cross'] = comb['q_freq'] * comb['a_freq']
    comb['q_freq_sq'] = comb['q_freq'] * comb['q_freq']
    comb['a_freq_sq'] = comb['a_freq'] * comb['a_freq']

    ret_cols = ['q_freq', 'a_freq', 'freq_mean',
                'freq_cross', 'q_freq_sq', 'a_freq_sq']

    return comb#[ret_cols]


