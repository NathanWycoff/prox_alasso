#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
import tensorflow_probability.substrates.jax as tfp
import matplotlib.colors as colors
import jax

seed = 1900
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

exec(open('python/jax_nsa.py').read())
exec(open('python/sim_lib.py').read())
exec(open('python/mosaic_nn_lib.py').read())

N = 425
P = 65

###
## Generate synthetic data that looks like the real thing.
# Generate "true" latent 2D coords from three gaussians.
y = np.repeat([0,1,2],np.array([367, 24, 34]))
mu_true = np.array([[0.,0.],[2.5,-4],[2, 2]])
SIGMA_true = np.array([[[1.,-0.95],[-0.95,1.]],[[1.,0.],[0.,1.]],[[1.,0.],[0.,1.]]])
Z = tfp.distributions.MultivariateNormalFullCovariance(mu_true[y,:], SIGMA_true[y,:,:]).sample(seed=key)
## Random loadings; only first 10 variables are involved.
R = 10
L = np.concat([np.random.normal(size=[2,R]), np.zeros([2,P-R])], axis = 1)
MU = Z @ L
X = tfp.distributions.Bernoulli(logits=MU).sample(seed=key)
###

## CV predictive error.
n_folds = 10
inds = np.arange(X.shape[0])
np.random.shuffle(inds)
folds = np.array_split(inds, n_folds)
nlls0 = np.zeros(n_folds)
nlls10 = np.zeros(n_folds)
for fi, fold in enumerate(folds):
    trainind = np.setdiff1d(np.arange(X.shape[0]), fold)
    X_train = X[trainind,:]
    y_train = y[trainind]
    X_test = X[fold,:]
    y_test = y[fold]

    params0, _ = train_net(X_train, y_train, 0.)
    params10, _ = train_net(X_train, y_train, 10.)

    preds0 = predict(params0, X_test.T).T
    nll0 = -np.mean(tfp.distributions.Categorical(logits = preds0).log_prob(y_test))
    nlls0[fi] = nll0

    preds10 = predict(params10, X_test.T).T
    nll10 = -np.mean(tfp.distributions.Categorical(logits = preds10).log_prob(y_test))
    nlls10[fi] = nll10

fig = plt.figure(figsize=[2,3])
plt.boxplot([nlls0,nlls10])
plt.xticks(ticks=[1,2],labels=[0,10])
plt.xlabel(r"$\tau$")
plt.ylabel('Neg Log Like')
plt.tight_layout()
plt.savefig("mosaic_cv.pdf")
plt.close()

## Plot with full data.
for tau0 in [0, 10]:
    params, costs = train_net(X, y, tau0)

    fig = plt.figure()
    plt.plot(costs)
    plt.savefig("costs.pdf")
    plt.close()

    Z = (params[0][0] @ X.T).T

    ng = 75
    rx = [np.min(Z[:,0]), np.max(Z[:,0])]
    gx = np.linspace(rx[0], rx[1], num = ng)
    ry = [np.min(Z[:,1]), np.max(Z[:,1])]
    gy = np.linspace(ry[0], ry[1], num = ng)

    G = np.zeros([ng*ng,2])
    for i in range(ng):
       for j in range(ng):
          G[i*ng+j,0] = gx[i]
          G[i*ng+j,1] = gy[j]

    def predict_latent(params, Z):
      activations = Z
      for w, b in params[1:-1]:
        outputs = w @ activations + b[:,jnp.newaxis]
        activations = relu(outputs)
      
      final_w, final_b = params[-1]
      logits = final_w @ activations + final_b[:,jnp.newaxis]
      return logits

    out = np.array(predict_latent(params, G.T).T)
    gclass = np.apply_along_axis(lambda x: np.argmax(x), 1, out)
    gcols = [['green','blue','red'][yi] for yi in gclass]


    ##### Left Column of Deep Active Classifier Figure
    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    fig = plt.figure(figsize=[4,4])

    plt.scatter(G[:,0],G[:,1], color = gcols, s = 10, marker = 's', alpha = 0.1)
    cols = [['green','blue','red'][yi] for yi in y]
    jitter = np.random.normal(size=Z.shape, scale = 2e-2)
    alpha = [0.6 if c =='green' else 1. for c in cols]
    plt.title(r"Deep Active Subspace; $\tau=$"+str(tau0))

    if analysis=='lowpenalty':
      patches = []
      patches.append(mpatches.Patch(color='green', label='Early Adopter'))
      patches.append(mpatches.Patch(color='blue', label='Vaccinated Skeptic'))
      patches.append(mpatches.Patch(color='red', label='Persistent Antivaxer'))
      plt.legend(handles=patches, prop = {'size':8}, framealpha = 1., loc = 'upper left')

    plt.scatter(Z[:,0]+jitter[:,0],Z[:,1]+jitter[:,1], color = cols, s = 5, alpha = alpha)
    plt.tight_layout()
    plt.savefig("proj_"+analysis+str(np.round(tau0,1))+".pdf")
    plt.close()
    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    print(params[0][0])
    BETA = params[0][0]

    K = 100 # Average over this many individuals when making inverse reg plots (Appendix C).

    ##### Right Column of Deep Active Classifier Figure
    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    fig = plt.figure(figsize=[5,4])
    lsize = 5

    #politicians = ['tedcruz','BarackObama','DonaldJTrumpJr','POTUS']
    politicians = [0,1,2]
    #entertainment = ['KendallJenner','Drake','SportsCenter','TheRealMikeEpps','NFL','KevinHart4real']
    entertainment = [3,4,5]
    #demographics = ['parent', 'race_Black', 'edred_<4y', 'POWNHOME_Rented']
    demographics = [6,7,8,9]
    plot_vars = politicians + entertainment + demographics 

    import matplotlib as mpl
    cm = mpl.colormaps['tab10']

    var_traj = {}
    for ci, coord in enumerate(['x','y']):
        gg = gx if coord=='x' else gy

        var_traj[coord] = pd.DataFrame(np.zeros([gg.shape[0], len(plot_vars)]))
        var_traj[coord].index = gg
        var_traj[coord].columns = plot_vars

        for i in range(gg.shape[0]):
            dists = np.square(Z[:,ci]-gg[i])
            idx = np.argpartition(dists, K)[:K]
            for v in plot_vars:
                var_traj[coord].loc[gg[i],v] = np.mean(X[idx,v])

    for v in plot_vars:
       mu_x = (np.mean(var_traj['x'][v]) + np.mean(var_traj['y'][v])) / 2
       sig_x = np.sqrt((np.var(var_traj['x'][v]) + np.var(var_traj['y'][v])) / 2)
       var_traj['x'].loc[:,v] = (var_traj['x'].loc[:,v] - mu_x) / sig_x
       var_traj['y'].loc[:,v] = (var_traj['y'].loc[:,v] - mu_x) / sig_x

    for ci, coord in enumerate(['x','y']):
        ii = 1 if ci==0 else 2
        plt.subplot(3,2,ii)
        plt.title("Political Figures " + coord.upper() + '-Axis')
        for vi,v in enumerate(politicians):
            plt.plot(var_traj[coord].loc[:,v], label = v, color = cm(vi/len(politicians)))

        ii = 3 if ci==0 else 4
        plt.subplot(3,2,ii)
        plt.title("Entertainment " + coord.upper() + '-Axis')
        for vi,v in enumerate(entertainment):
            plt.plot(var_traj[coord].loc[:,v], label = v, color = cm(vi/len(entertainment)))

        ii = 5 if ci==0 else 6
        plt.subplot(3,2,ii)
        plt.title("Demographics " + coord.upper() + '-Axis')
        for vi,v in enumerate(demographics):
            plt.plot(var_traj[coord].loc[:,v], label = v, color = cm(vi/len(demographics)))

    plt.tight_layout()
    plt.savefig('inv_reg_'+analysis+str(np.round(tau0,1))+'.pdf')
    plt.close()


    fig = plt.figure(figsize=[1.8,4])

    for ti,target in enumerate([politicians, entertainment, demographics]):
      plt.subplot(3,1,ti+1)
      patches = []
      for vi,v in enumerate(target):
        patches.append(mpatches.Patch(color=cm(vi/len(target)), label=v))
      plt.legend(loc=['upper left','center left','lower left'][ti],handles=patches, prop = {'size':8})

      plt.axis("off")

    plt.tight_layout()
    plt.savefig('mosaic_nn_legend.pdf')
    plt.close()
    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

