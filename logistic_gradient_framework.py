#!/usr/bin/python
from scipy.stats import truncnorm,uniform
import numpy as np
import itertools
import copy, pickle, os, time
import sys


####################################################################################
#Helper functions
####################################################################################
file_index=0
def reportError(msg):
    global key1;global key2;global indexx;
    os.system("echo \""+str(key1)+"-"+str(key2)+": "+msg+"\">> errorsExperiment"+str(indexx))

with open("data/corr_all_key", 'rb') as fileOpen:
    corr = pickle.load(fileOpen)

def getKeyPair():
    numKey=1009
    keyPair=[]
    for key1,key2 in itertools.product(range(numKey),range(numKey)):
        if key2<=key1 or corr[key1][key2]<2: continue;
        keyPair.append([key1,key2])
    def getKey(keyPair):
        global corr
        return int(corr[keyPair[0]][keyPair[1]])
    keyPair=sorted(keyPair,key=getKey,reverse=True)
    print("len", len(keyPair))
    return keyPair

####################################################################################
# Functions to run experiments
####################################################################################
def main(thread_index):
    start_time = time.time();

    numKey=1009
    numCores=14
    lowVarAdvKeyPair = pickle.load(open("data/lowVarAdv", 'rb'))
    with open("data/corr_all_key", 'rb') as fileOpen:
        corr = pickle.load(fileOpen)

    lowVarAdv  = [set() for i in range(numKey)]
    for pair in lowVarAdvKeyPair:
        lowVarAdv[pair[1]].add(pair[0])

    advKeyPair = [set() for i in range(numKey)]
    for key1 in range(numKey):
        if key1%100==0: print(key1)
        for key2 in range(numKey):
            if(key2 > key1) and corr[key1][key2]>1:
                with open("data/keys-"+str(key1)+"-"+str(key2)+"/advertiser", 'rb') as fileOpen:
                    sharedAdvertisers = pickle.load(fileOpen)
                for adv in sharedAdvertisers:
                    if adv not in lowVarAdv[key1] and adv not in lowVarAdv[key2]:
                        advKeyPair[key1].add(adv)
                        advKeyPair[key2].add(adv)

    ijk = 0
    cnt=0
    for key in range(len(advKeyPair)):
        if len(advKeyPair[key])<2: continue;
        for adv in advKeyPair[key]:
            ijk+=1
            ## Issues with range of phi_min or phi_max in these keys
            if ijk%numCores == thread_index:
                print(key,adv)
                generate_pdf_cdf_arrays(key,adv)
    print("Total=",cnt)

def generate_pdf_cdf_arrays(key,adv):
    ## For error reporting
    start=time.time()
    ##################################################
    ## Get distributions of advertisers
    ##################################################

    folder="data/keys-"+str(key)+"-adv"+str(adv)+"/"
    func_cdf = pickle.load(open(folder+"cdf", 'rb'))
    func_pdf = pickle.load(open(folder+"pdf", 'rb'))
    func_inv_cdf = pickle.load(open(folder+"inv_cdf", 'rb'))

    # shift=np.zeros((numAttr,len(adv)))
    samples=60000
    min_x = -30; max_x =  100
    def si(s): return int(samples * s/(max_x-min_x));

    # x=np.linspace(min_x,max_x,samples)
    # y=[[adv[i].dist[a].pdf(x) for i in range(len(adv))] for a in range(numAttr)]
    # z=[[adv[i].dist[a].cdf(x) for i in range(len(adv))] for a in range(numAttr)]

    x=np.linspace(min_x,max_x,samples)

    def get_pdf_cdf_arrays(func_cdf,func_pdf,func_inv_cdf):
        iter=100*samples
        # bid = np.linspace(min_x,max_x,iter)

        r = np.random.rand(int(iter))*0.998+0.001
        bid = func_inv_cdf[0](r)

        one=np.ones(int(iter))
        virBid = bid-(one-r)/(func_pdf[0](bid)+(one/100000.0))

        bins = np.linspace(-30,100,samples)
        digitized = np.digitize(virBid, bins)
        pdf = np.array([0 for i in range(len(bins))])
        for i in range(len(digitized)):
            if digitized[i] != samples: pdf[digitized[i]]+=1
        pdf[0]=0 # remove lower tail
        pdf = (pdf / np.sum(pdf)) * (60000/130.0)
        cdf=np.zeros(len(pdf))
        cdf[1:]=np.cumsum(pdf[1:]*np.diff(bins))
        pdf=pdf/cdf[-1]
        cdf=cdf/cdf[-1]

        return pdf,cdf

    y,z = get_pdf_cdf_arrays(func_cdf,func_pdf,func_inv_cdf);

    folder="data/keys-"+str(key)+"-adv"+str(adv)+"/"
    with open(folder+"pdf_array_virtual_valuation", 'wb') as file:
        pickle.dump(y, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+"cdf_array_virtual_valuation", 'wb') as file:
        pickle.dump(z, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(time.time()-start,"sec")


if __name__ == '__main__' :
    main()
