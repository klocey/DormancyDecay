from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import sys
from os.path import expanduser
#import statsmodels.api as sm


def xfrm(X, _max): return _max-np.array(X)

def figplot(dat, y, x, seed, xlab, ylab, fig, fit, n):

    fs, sz = 8, 1
    a = 1.0

    e = max(seed)
    dat = dat.tolist()
    y = y.tolist()
    #x = x**2
    x = x.tolist()
    seed = seed.tolist()

    fig.add_subplot(2, 2, n)

    clrs = []

    for i, val in enumerate(dat):
        sd = seed[i]
        clr = str()
        if sd <= e*0.3: clr = 'red'
        elif sd < e*0.4: clr = 'orange'
        elif sd < e*0.5: clr = 'yellow'
        elif sd < e*0.6: clr = 'lawngreen'
        elif sd < e*0.7: clr = 'green'
        elif sd < e*0.8: clr = 'deepskyblue'
        elif sd < e*0.9: clr = 'blue'
        else: clr = 'purple'
        clrs.append(clr)

    plt.scatter(x, y, s = sz, c=clrs, linewidths=0.0, alpha=a, edgecolor=None)
    plt.xlabel(xlab, fontsize=fs+2)
    plt.ylabel(ylab, fontsize=fs+2)
    plt.tick_params(axis='both', labelsize=fs)
    return fig



def figfunction(met1, met2, fname, disp, label):

    ws, hs = 0.4, 0.4
    mydir = expanduser("~/GitHub/DormancyDecay")

    fit = 1
    df = pd.read_csv(mydir+'/model/ModelData/modelresults.txt')
    
    #i_ls = list(set(df['Sim']))
    #print(max(i_ls))
    #sys.exit()
    
    df = df[df['disperse'] == disp]
    
    tot = df.shape[0]
    df = df[df['fit'] == 1]
    fits = df.shape[0]
    
    if met1 == 'bray' and disp == 0:
        print('No dispersal:', 100*fits/tot)
    elif met1 == 'bray' and disp == 1:
        print('Dispersal:', 100*fits/tot)
        
    fig = plt.figure()

    if met2 == 'p_err': ylab = 'Percent error'
    elif met2 == 'p_dif': ylab = 'Percent difference'
    elif met2 == 'a_dif': ylab = 'Difference'

    xlab = 'Environmental filtering'
    
    
    if label == 'avg':
        y = df[met1 + '_' + labels[0] + '-' + met2]
        labels2 = ['e_allslope','g_actslope','g_allslope']
        for l in labels2:
            y += df[met1 + '_' + l + '-' + met2]
        y = y/4
    else:
        y = df[met1 + '_' + label + '-' + met2]
        
        
    fig = figplot(df['fit'], y, df['env_r'], df['env_r'], xlab, ylab, fig, fit, 1)

    xlab = 'Dormant death'
    fig = figplot(df['fit'], y, df['dded'], df['env_r'], xlab, ylab, fig, fit, 2)

    if disp == 1:
        xlab = 'Active dispersal'
        fig = figplot(df['fit'], y, df['ad_s'], df['env_r'], xlab, ylab, fig, fit, 3)

        xlab = 'Dormant dispersal'
        fig = figplot(df['fit'], y, df['dd_s'], df['env_r'], xlab, ylab, fig, fit, 4)
        
        
    #### Final Format and Save #####################################################
    plt.subplots_adjust(wspace=ws, hspace=hs)
    #plt.savefig(mydir+'/figs/FromSims/m1/'+label+'/'+met1+'-'+met2+fname+'.png',
    #   dpi=400, bbox_inches = "tight")
    
    plt.savefig(mydir+'/figs/FromSims/temp/'+label+'/'+met1+'-'+met2+fname+'.png',
        dpi=400, bbox_inches = "tight")
    plt.close()


    




fnames = ['_no-dispersal', '_dispersal']
disperse = [0, 1]

metrics1 = ['bray', 'sore', 'canb']
metrics2 = ['p_err']

labels = ['e_actslope','e_allslope','g_actslope','g_allslope','avg']
for label in labels:
    for i, fname in enumerate(fnames):
        for met1 in metrics1:
            for met2 in metrics2:
        
                disp = i
                figfunction(met1, met2, fname, disp, label)
