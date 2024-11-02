#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.packages import importr
glmnet = importr("glmnet")

test_funcs = {}
r = robjects.r

exec(open('python/sim_lib.py').read())

def glmnet_fit(X, y, XX, lik, taus = None):
    family = None
    if lik == 'normal':
        family = 'gaussian'
    elif lik == 'poisson':
        family = 'poisson'
    elif lik == 'bernoulli':
        family = 'binomial'
    else:
        raise Exception("Unkown family in glmnet_fit.")

    X_arr = robjects.FloatVector(X.T.flatten())
    X_R = robjects.r['matrix'](X_arr, nrow = X.shape[0])

    XX_arr = robjects.FloatVector(XX.T.flatten())
    XX_R = robjects.r['matrix'](XX_arr, nrow = XX.shape[0])

    y_arr = robjects.FloatVector(y)
    y_R = robjects.r['matrix'](y_arr, ncol = 1)

    if taus is None:
        fit = glmnet.cv_glmnet(X_R, y_R, family = family)
        betahat = r['as.matrix'](r['coef'](fit))[1:].flatten()
        preds = r['predict'](fit, XX_R).flatten()
    else:
        fit = glmnet.glmnet(X_R, y_R, family = family, alpha = 1., **{'lambda' : taus})
        betahat = r['as.matrix'](r['coef'](fit))[1:,:].T
        preds = r['predict'](fit, XX_R).T
    del fit, X_arr, X_R, XX_arr, XX_R, y_arr, y_R
    r['gc']()
    r['gc']()
    return betahat, preds
