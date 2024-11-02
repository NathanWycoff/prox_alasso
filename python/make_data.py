#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  form_kin40.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.11.2024

import numpy as np
import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

exec(open("python/sim_lib.py").read())

if not os.path.exists(staging_dir):
    os.makedirs(staging_dir)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for ds in datasets_to_use:
    if ds =='abalone':
        uds = fetch_ucirepo(id=1)
        X = uds['data']['features']
        Xc = pd.get_dummies(X['Sex'], drop_first = True).astype(float)
        Xnc = X.drop('Sex', axis = 1)
        X = np.array(pd.concat([Xc, Xnc],axis=1))
        y = np.array(uds['data']['targets']['Rings']).flatten()
    elif ds =='obesity':
        uds = fetch_ucirepo(id=544)

        Xdf = uds['data']['features']
        cat = ['Gender', 'family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']
        Xdfcat = pd.get_dummies(Xdf.loc[:,cat], drop_first = True).astype(float)
        X = np.array(pd.concat([Xdf.drop(cat,axis=1), Xdfcat], axis = 1))
        od = {
                'Insufficient_Weight' : 0,
                'Normal_Weight' : 1,
                'Overweight_Level_I' : 2,
                'Overweight_Level_II' : 3,
                'Obesity_Type_I' : 4,
                'Obesity_Type_II' : 5,
                'Obesity_Type_III' : 6,
                }
        y = np.array(uds['data']['targets'].map(lambda x: od[x])).flatten()
    elif ds=='parkinsons':
        uds = fetch_ucirepo(id=189)
        X = np.array(uds['data']['features'])
        y = np.array(uds['data']['targets']['motor_UPDRS']).flatten()
    elif ds=='aids':
        uds = fetch_ucirepo(id=890)
        X = np.array(uds['data']['features'])
        y = np.array(uds['data']['targets']).flatten()
    elif ds=='rice':
        uds = fetch_ucirepo(id=545)
        X = np.array(uds['data']['features'])
        y = np.array(pd.get_dummies(uds['data']['targets'],drop_first=True)).flatten().astype(float)
    elif ds=='spam':
        uds = fetch_ucirepo(id=94)
        X = np.array(uds['data']['features'])
        y = np.array(uds['data']['targets']).flatten().astype(float)
    elif ds=='dropout':
        uds = fetch_ucirepo(id=697)
        X = np.array(uds['data']['features'])
        y = np.array(uds['data']['targets']=='Dropout').flatten().astype(float)
    elif ds=='shop':
        uds = fetch_ucirepo(id=468)
        cat = ['Month','VisitorType','Weekend']
        Xdf = uds['data']['features']
        Xdfcat = pd.get_dummies(Xdf.loc[:,cat], drop_first = True).astype(float)
        X = np.array(pd.concat([Xdf.drop(cat,axis=1), Xdfcat], axis = 1))
        y = np.array(pd.get_dummies(uds['data']['targets'],drop_first=True)).flatten().astype(float)
    else:
        raise Exception("Unknown Dataset!")

    df = pd.DataFrame(X)
    df.columns = ['X'+str(i) for i in range(X.shape[1])]
    df['y'] = y

    eps = 1e-6
    if ds in class_problems:
        df['y'] = (df['y']-min(df['y'])) / (max(df['y'])-min(df['y']))
    elif ds in reg_problems:
        df['y'] = (df['y'] - np.mean(df['y'])) / (np.std(df['y'])+eps)
    else:
        raise Exception("Unknown dataset type!")
    for i in range(df.shape[1]-1):
        df.iloc[:,i] = (df.iloc[:,i] - np.mean(df.iloc[:,i])) / (np.std(df.iloc[:,i])+eps)

    outname = data_dir+ds+'.csv'
    df.to_csv(outname, index = False)

