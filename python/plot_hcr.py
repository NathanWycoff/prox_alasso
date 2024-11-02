#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  plot_hcr2.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 02.09.2024

import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.signal import medfilt

exec(open('python/hcr_settings.py').read())

with open("pickles/traj_hcr_"+str(eu_only)+'_'+expansion+'.pkl', 'rb') as f:
    df_means, df_zeros, resdf = pickle.load(f)

for i in range(len(df_means)):
    df_means[i].columns = [np.flip(tau_range)[i], 'name']
    df_means[i]['name'] = [x.replace('best_est','deaths') for x in df_means[i]['name']]
for i in range(len(df_zeros)):
    df_zeros[i].columns = [np.flip(tau_range)[i], 'name']
    df_zeros[i]['name'] = [x.replace('best_est','deaths') for x in df_zeros[i]['name']]

res = df_means[0]
res0 = df_zeros[0]
for i in range(1,len(df_means)):
    res = pd.merge(res,df_means[i], how = 'outer', on = 'name')
    res0 = pd.merge(res0,df_zeros[i], how = 'outer', on = 'name')

res.index = res.name
res = res.drop('name',axis=1)
res = res.fillna(0)
res0.index = res0.name
res0 = res0.drop('name',axis=1)
res0 = res0.fillna(0)

if res.shape[1] > 40:
    medrad = 15
else:
    medrad = 3
for ii,v in enumerate(res.index):
    res.loc[v,:] =  medfilt(res.loc[v,:].astype(float),medrad)
for ii,v in enumerate(res0.index):
    res0.loc[v,:] =  medfilt(res0.loc[v,:].astype(float),medrad)

# To aid visualization keep everthing close.
thresh = 100
res = np.maximum(-thresh, np.minimum(thresh, res))
res0 = np.maximum(-thresh, np.minimum(thresh, res0))

##### Get first nonzero and order accordingly
first_nz = np.zeros(res.shape[0]).astype(int)
for i,v in enumerate(res.index):
    if np.any(res.loc[v,:]!=0):
        first_nz[i] = np.where(res.loc[v,:] != 0)[0][0]
    else:
        first_nz[i] = res.shape[1]-1
label_K = 5
first_inds = np.argpartition(-first_nz, -label_K)[-label_K:]
first_nz[first_inds]
order = [x for _, x in sorted(zip(first_nz, np.arange(res.shape[0])))]
res = res.iloc[order,:]

first_nz0 = np.zeros(res0.shape[0]).astype(int)
for i,v in enumerate(res0.index):
    if np.any(res0.loc[v,:]!=0):
        first_nz0[i] = np.where(res0.loc[v,:] != 0)[0][0]
    else:
        first_nz0[i] = res0.shape[1]-1
label_K = 5
first_inds = np.argpartition(-first_nz0, -label_K)[-label_K:]
first_nz0[first_inds]
order = [x for _, x in sorted(zip(first_nz0, np.arange(res0.shape[0])))]
res0 = res0.iloc[order,:]
##### Get first nonzero and order accordingly

if not synthetic:
    xlim = 80
else:
    xlim = res.shape[1]

res = res.iloc[:,1:xlim]
res0 = res0.iloc[:,1:xlim]

fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=[5,2])

if not synthetic:
    vline = 34
else:
    vline = int(np.round(res.shape[1]*0.3))

cnt = 0
if not synthetic:
    nbigm = 4
else:
    nbigm = 3
cm =  mpl.colormaps['tab20b']
topcol = [cm(i/(nbigm-1)) for i in range(nbigm)]
for ii,v in enumerate(res.index):
    vs = res.loc[v,:]
    indmax = np.argmax(np.abs(vs.iloc[1:xlim]))
    if np.any(vs!=0) and np.where(vs!=0)[0][0] <= xlim:
        if cnt < nbigm:
            col = topcol[cnt]
            label = v
        else:
            col = 'gray'
            label = None
        cnt += 1
    else:
        label = None
        col = 'gray'
    a0.plot(res.columns, res.loc[v,:], label = label, color = col)[0]
a0.legend(prop={'size':5}, loc = 'upper right', framealpha=1.)
ll, ul = a0.get_ylim()
a0.vlines(resdf['tau'][vline], ll, ul, linestyle='--', color = 'gray')
a0.set_xscale('log')
a0.set_title("Hurdle Model Coefficient Trajectory", fontsize=8)
a0.set_ylabel("Coefficent Estimates", fontsize = 8)
a0.set_xlabel(r"$\tau$")
a0.tick_params(axis='both', which='major', labelsize=5)
a0.tick_params(axis='both', which='minor', labelsize=5)

nll = medfilt(resdf['nll'],medrad)
taus = resdf['tau']
a1.plot(taus[1:xlim], nll[1:xlim])
a1.set_xscale('log')
a1.set_title("Predictive NLL", fontsize = 8)
a1.set_xlabel(r"$\tau$")
a1.set_ylim([np.min(nll), np.max(nll[1:xlim])])
a1.tick_params(axis='both', which='major', labelsize=5)
a1.tick_params(axis='both', which='minor', labelsize=5)
a1.vlines(resdf['tau'][vline], np.min(nll), np.max(nll[1:xlim]), linestyle='--', color = 'gray')
a1.set_xlim([np.min(taus[1:xlim]), np.max(taus[1:xlim])])

plt.tight_layout()
plt.savefig('traj'+str(eu_only)+'_'+expansion+'.pdf')
plt.close()
