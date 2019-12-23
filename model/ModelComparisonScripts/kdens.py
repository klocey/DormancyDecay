from __future__ import division
import  matplotlib.pyplot as plt
import pandas as pd
from os.path import expanduser



def figplot(cts, xlab, ylab, fig, n, lab, minx):

        fs = 8

        ax = fig.add_subplot(2, 2, n)
        
        yls = list()
        xls = list()
        xs = range(minx, 51)
        for x in xs:
            ct2 = cts.count(x)
            p = 100*(ct2)/len(cts)
            #if x == 50: print(p)
            if p > 0 and x > 40:
                xls.append(x)
                yls.append(p)
            
            
        
        plt.scatter(xls, yls, color = '0.7', linewidths=1, edgecolor='k')

        plt.xlabel(xlab, fontsize=fs)
        plt.ylabel(ylab, fontsize=fs)
        plt.xlim(min(xls)-0.5, 50.5)
        plt.tick_params(axis='both', labelsize=fs)
        
        x1 = list(xls)
        ax.set_xticks(x1)
        ax.set_xticklabels(x1, minor=False)
        return fig




#p, fr, _lw, w, fs, sz = 2, 0.25, 0.5, 1, 6, 4
ws, hs = 0.45, 0.5

mydir = expanduser("~/GitHub/DormancyDecay")

df = pd.read_csv(mydir+'/model/ModelData/modelresults-numfit.txt')
fig = plt.figure()

xlab = 'No. aspects reproduced\nfrom empirical data'


ylab = '% of models that\ninclude dispersal'
df2 = df[df['disperse'] == 1]
x = df2['numfits'].tolist()
minx = min(x)
lab = 'With dispersal'
fig = figplot(x, xlab, ylab, fig, 1, lab, minx)


ylab = '% of models that\nexclude dispersal'
df2 = df[df['disperse'] == 0]
x = df2['numfits'].tolist()
minx = min(x)
lab = 'No dispersal'
fig = figplot(x, xlab, ylab, fig, 2, lab, minx)


f1 = df[df['numfits'] == 50]
f2 = len(f1[f1['disperse'] == 0])
print(100*f2/len(f1), '% of fit models had no dispersal')

print(100*len(f1)/len(df), '% of total models that were fit')

#### Final Format and Save #####################################################
plt.subplots_adjust(wspace=ws, hspace=hs)
plt.savefig(mydir+'/figs/FromSims/kdens.png', dpi=400, bbox_inches = "tight")
plt.close()