#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  sim_lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.19.2023

import numpy as np

## Define simulations to run
sims = ['synthetic','uci']
sparsity_types = ['random','group','hier2nd']

reg_problems = ['abalone','obesity','parkinsons']
class_problems = ['aids','rice','spam','dropout','shop']

datasets_to_use = reg_problems + class_problems

likelihoods = ['normal','nb','cauchy','bernoulli']

Ns = [10000]
iters = 30

data_dir = "./data/"
staging_dir = 'staging/'


## Some helper functions
def random_sparsity(P,Pnz):
    nonzero = np.zeros(P)
    nonzero[np.random.choice(P,Pnz)] = 1.

    return nonzero

def group_sparsity(Pu,Pnz):
    nonzero = np.zeros(P)
    nonzero[np.random.choice(P,Pnz)] = 1.

    return nonzero

# The matrix Gamma links main effects to the interaction terms including them.
# Each column, of which there are P, gives indices of the interaction terms which it belongs to.
def make_gamma_mat(P):
    tops = []
    bots = []
    for i in range(P):
        if i > 0:
            incs = np.concatenate([[0],np.cumsum(P-1-np.arange(1,i))])
            tops.append(i+incs)
        else:
            tops.append([])
    for i in range(P-1):
        if i >0:
            startat = tops[i+1][i]
        else:
            startat = 1
        bots.append(np.arange(startat,startat+P-1-i))
    bots.append([])
    gammat = np.zeros([P-1,P])
    for i in range(P):
        gammat[:,i] = np.concatenate([tops[i], bots[i]])
    gammat -= 1
    return gammat

def add_int(Xu, var_names = None):
    N,Pu = Xu.shape

    if var_names is None:
        var_names = ['X'+str(i) for i in range(Pu)]

    Pi = int(scipy.special.binom(Pu,2))
    P = Pu + Pi 

    # Interactions
    Xi = np.zeros([N,Pi])
    ind = 0
    int_name = []
    for i in range(Pu-1):
        for j in range(i+1,Pu):
            Xi[:,ind] = Xu[:,i]*Xu[:,j]
            ind += 1
            int_name.append(var_names[i]+'-'+var_names[j])

    X = np.concatenate([Xu,Xi,], axis = 1)
    Xdf = pd.DataFrame(X)
    Xdf.columns = var_names + int_name 

    return Xdf

def add_int_quad(Xu, var_names = None):
    N,Pu = Xu.shape

    if var_names is None:
        var_names = ['X'+str(i) for i in range(Pu)]

    Pi = int(scipy.special.binom(Pu,2))
    Pq = Pu
    P = Pu + Pi + Pq

    # Interactions
    Xi = np.zeros([N,Pi])
    ind = 0
    int_name = []
    for i in range(Pu-1):
        for j in range(i+1,Pu):
            Xi[:,ind] = Xu[:,i]*Xu[:,j]
            ind += 1
            int_name.append(var_names[i]+'-'+var_names[j])

    # Quadratics
    Xq = np.zeros([N,Pq])
    ind = 0
    quad_name = []
    for i in range(Pu):
        Xq[:,i] = np.square(Xu[:,i])
        quad_name.append(var_names[i]+'^2')

    X = np.concatenate([Xu,Xi,Xq], axis = 1)
    Xdf = pd.DataFrame(X)
    Xdf.columns = var_names + int_name + quad_name

    return Xdf


def hier2nd_sparsity(Pu,Pnz):
    ngroups = int(scipy.special.binom(Pu,2))
    P = 2*Pu + ngroups

    group_sparsity = np.zeros(ngroups)
    group_sparsity[np.random.choice(ngroups,Pnz)] = 1.
    int_sparsity = group_sparsity

    GG = make_gamma_mat(Pu).astype(int)
    GAMMA = group_sparsity[GG]
    me_sparsity = np.any(GAMMA, axis = 0).astype(float)
    q_sparsity = me_sparsity

    v1 = np.zeros(ngroups).astype(int)
    v2 = np.zeros(ngroups).astype(int)
    ii = 0
    for i in range(Pu-1):
        for j in range(i+1,Pu):
            v1[ii] = i
            v2[ii] = j
            ii += 1

    nonzero = np.concatenate([me_sparsity, int_sparsity, q_sparsity])

    return nonzero, ngroups, P, v1, v2
