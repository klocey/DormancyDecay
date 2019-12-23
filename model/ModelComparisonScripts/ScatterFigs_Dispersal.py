from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from os.path import expanduser


mydir = expanduser("~/GitHub/DormancyDecay")


def figplot(dat, y, x, seed, xlab, ylab, fig, n):

    fs, sz, a = 6, 0.1, 1
    e = max(seed)
    dat = dat.tolist()
    y = y.tolist()
    
    x = x.tolist()
    seed = seed.tolist()

    fig.add_subplot(3, 3, n)

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
    plt.xlabel(xlab, fontsize=fs)
    plt.ylabel(ylab, fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs)
    
    if n == 1:
        plt.text(-.6, 3, 'Bray-Curtis', fontsize=fs+2, rotation=90)
    elif n == 4:
        plt.text(-.6, 400, 'Sorensen', fontsize=fs+2, rotation=90)
    elif n == 7:
        plt.text(-.6, 100, 'Canberra', fontsize=fs+2, rotation=90)
    
    return fig



def figfunction(fig, met1, label, n, df):

    met2 = 'p_err'    
    ylab = 'Percent error'
    xlab = 'Active dispersal'
    
    y = str()
    if label == 'avg':
        y = df[met1 + '_' + 'e_actslope' + '-' + met2]
        labels2 = ['e_allslope','g_actslope','g_allslope']
        for l in labels2:
            y += df[met1 + '_' + l + '-' + met2]
        y = y/4
    else:
        y = df[met1 + '_' + label + '-' + met2]
    
    fig = figplot(df['fit'], y, df['ad_s'], df['env_r'], xlab, ylab, fig, n)

    xlab = 'Dormant dispersal'
    fig = figplot(df['fit'], y, df['dd_s'], df['env_r'], xlab, ylab, fig, n+1)
        
    return fig
    


    

for i in range(2):
    mydir = expanduser("~/GitHub/DormancyDecay")
    df = pd.read_csv(mydir+'/model/ModelData/modelresults-numfit.txt')
    df = df[df['disperse'] == i]
    
    tot = df.shape[0]
    df = df[df['fit'] == 1]
    fits = df.shape[0]
    
    y1 = df['bray' + '_' + 'e_allslope' + '-' + 'p_err']
    y1 = len(y1)
    
    y2 = df['bray' + '_' + 'e_allslope' + '-' + 'p_err']
    labels2 = ['e_allslope','g_actslope','g_allslope']
    for l in labels2:
        y2 += df['bray' + '_' + l + '-' + 'p_err']
    y2 = y2/4
    
    y2s = []
    for y in y2:
        if y < 10.0: 
            y2s.append(y)
    #y2 = df[df['canb' + '_' + 'e_allslope' + '-' + 'p_err'] < 10]
    y2 = len(y2s)
    
    print('percent closer than 10%:', 100 * y2/y1)
    
    if i == 0:
        print('No dispersal:', 100*fits/tot)
    elif i == 1:
        print('Dispersal:', 100*fits/tot)
        
    print('AvgAct:', np.mean(df['avgAct']), 'AvgAll', np.mean(df['avgAll']))
    print('Sact:', np.mean(df['Sact']), 'Sall:', np.mean(df['Sall']),'\n')
    #sys.exit()


  

df = pd.read_csv(mydir+'/model/ModelData/modelresults-numfit.txt')  
df = df[df['fit'] == 1]
df = df[df['disperse'] == 0]

    
metrics = ['bray', 'sore', 'canb']
labels = ['e_allslope'] #, 'e_actslope', 'g_actslope', 'g_allslope', 'avg']

for label in labels:
    for met1 in metrics:  
        fig = plt.figure()
        ns = [1,4,7]
        for i, met1 in enumerate(metrics):
            n = ns[i]
            fig = figfunction(fig, met1, label, n, df)

        ws, hs = 0.5, 0.5
        plt.subplots_adjust(wspace=ws, hspace=hs)
        plt.savefig(mydir+'/figs/FromSims/temp/2x3'+label+'-dispersal.png', dpi=400, bbox_inches = "tight")
        plt.close()