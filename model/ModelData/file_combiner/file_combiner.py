import shutil
import os

mydir = os.path.expanduser("~/GitHub/DormancyDecay/model/ModelData")


with open(mydir+'/modelresults-numfit.txt','wb') as wfd:
    for i in range(1, 102):
        f = '/small_files/f'+str(i)+'.txt'
        with open(mydir+f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
            