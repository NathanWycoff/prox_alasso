#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys
import os
import glob

exec(open('python/sim_lib.py').read())

mode = sys.argv[1]

if mode=='small':
    dd = []
    for sim in sims:
        for sparsity_type in sparsity_types:
            if sim=='uci' and sparsity_type!='hier2nd':
                continue
            exec(open('python/sim_settings.py').read())
            if sim=='synthetic':
                ds = range(len(likelihoods)*len(Ns))
            else:
                ds = datasets_to_use

            for s_i in ds:
                for seed in range(iters):
                    dd.append(sim + ' ' + sparsity_type + ' ' + str(s_i)+' '+str(seed))

            folder = 'sim_out/'+simout_dir
            if os.path.isdir(folder):
                files = glob.glob(folder+"/*")
                for f in files:
                    os.remove(f)
            else:
                os.makedirs(folder)

    pd.DataFrame(dd).to_csv('sim_args.txt', index = False, header=False)
elif mode=='hcr':
    exec(open('python/hcr_settings.py').read())
else:
    raise Exception("Unknown mode arg to clean_dir!")



# Make debug folder if not already made.
folder = 'debug_out'
if not os.path.isdir(folder):
    os.makedirs(folder)

