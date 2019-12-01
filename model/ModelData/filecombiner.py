import shutil
import os

mydir = os.path.expanduser("~/GitHub/DormancyDecay/model/ModelData")


with open('modelresults.txt','wb') as wfd:
    for f in ['f1.txt','f2.txt','f3.txt','f4.txt','f5.txt','f6.txt','f7.txt',
              'f8.txt','f9.txt','f10.txt','f11.txt']:
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd)
            