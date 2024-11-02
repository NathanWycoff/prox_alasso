#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  plot_ada_sim.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.18.2023

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import sys

sim = sys.argv[1]
sparsity_type = sys.argv[2]

exec(open('python/sim_lib.py').read())
exec(open('python/sim_settings.py').read())

import glob
files = glob.glob('sim_out/'+sim+"_"+sparsity_type+"/*")
dfs = []
for f in files:
    dfs.append(pd.read_csv(f))
res = pd.concat(dfs)


if sim=='uci':
    # NOTE: Change which line is commented here to determine which of the two plots gets made. 
    # NOTE: Sorry.
    uci_second = False
    #uci_second = True
    set_inds = np.sort(list(set(res['Setting'])))
    if uci_second:
        keepers = set_inds[len(set_inds)//2:]
    else:
        keepers = set_inds[:len(set_inds)//2]
    res = res.loc[res['Setting'].isin(keepers),:]

methods = [x for x in model_colors.keys() if x in models2try]
ncomp = len(methods)

isjags = res['Method']=='jags'
isnan = res['yy-MSE']==1.0
res.loc[np.logical_and(isjags, isnan),'yy-MSE'] = np.nan
res.loc[np.logical_and(isjags, isnan),'Time'] = np.nan

keep = np.logical_and(res['Method']=='sbl_group', res['Setting']==3.0)
res.loc[keep,:]

cols = [model_colors[x] for x in methods]

set_inds = np.sort(list(set(res['Setting'])))
S = len(set_inds)

fig = plt.figure(figsize=[10,6]) # To fit in article
for ti,targ in enumerate(['MSE','Time']):
    for s_i, s in enumerate(set_inds):
        ind = S*ti+s_i
        plt.subplot(2,len(set_inds),ind+1)
        msecol = 'beta-MSE' if sim=='synthetic' else 'yy-MSE'
        if targ=='MSE':
            msecol_title = r" $\beta$-MSE" if sim=='synthetic' else (' Misclass Rate' if set_inds[s_i] in class_problems else ' MSE')
            subsets = [res.loc[(res['Method']==meth)*(res['Setting']==s), msecol] for meth in methods]
            title = settings[s_i][3].title()+msecol_title if sim=='synthetic' else set_inds[s_i].title() + msecol_title
        elif targ == 'Time':
            subsets_mse = [res.loc[(res['Method']==meth)*(res['Setting']==s),msecol] for meth in methods]
            subsets = [res.loc[(res['Method']==meth)*(res['Setting']==s),'Time'] for meth in methods]
            for si in range(len(subsets_mse)):
                subsets[si][np.isnan(subsets_mse[si])] = np.nan
            title = settings[s_i][3].title()+ " Wall Time (s)" if sim=='synthetic' else set_inds[s_i].title() + " Wall Time (s)"
        else:
            raise Exception

        subsets = [[x for x in ss if not np.isnan(x)] for ss in subsets]

        bplot = plt.boxplot(subsets, positions = [ncomp*s_i+i for i in range(len(methods))],patch_artist=True, medianprops=dict(color="black"), widths = 0.85)
        for patch, color in zip(bplot['boxes'], cols):
            patch.set(color=color)
            patch.set(facecolor = color)

        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_xticks([])
        ll = ax.get_ylim()
        if not (sim=='uci' and targ=='MSE'):
            plt.yscale('log')
        if targ == 'MSE':
            plt.title(title, fontdict={'weight':'bold'})
        elif targ == 'Time':
            plt.title(title, fontdict={'weight':'bold','size':10})
        else:
            raise Exception

plt.tight_layout()
fname = sim+'_'+sparsity_type+"_sim.pdf" if sim=='synthetic' else sim+'_'+sparsity_type+str(int(uci_second))+"_sim.pdf"
plt.savefig(fname)
plt.close()

linewidth = 0.15
lines = []
labels = []
for i,m in enumerate(methods):
    lines.append(plt.hlines(0,i,i+linewidth, color = model_colors[m], linewidth=7))
    if m in nice_names.keys():
        title = nice_names[m]
    else:
        title = m
    labels.append(title)
fig = plt.figure(figsize=[10,0.5])
plt.legend(lines, labels, ncol = len(lines), prop = {'weight':'bold', 'size':10}, loc = 'center')
plt.axis('off')
plt.tight_layout()
plt.savefig(sim+'_'+sparsity_type+"_legend.pdf")
plt.close()
