#!/usr/bin/python
from scipy.stats import truncnorm,uniform
import numpy as np
import itertools
import copy, pickle, os, time
import sys


####################################################################################
#Helper functions
####################################################################################
f_index=0
def report_error(msg):
    global key1;global key2;global f_index;
    os.system("echo \""+str(key1)+"-"+str(key2)+": "+msg+"\">> errorsExperiment"+str(f_index))

with open("data/corr_all_key", 'rb') as f:
    corr = pickle.load(f)

def getKeyPair():
    number_of_keys=1009
    key_pair=[]
    for key1,key2 in itertools.product(range(number_of_keys),range(number_of_keys)):
        if key2<=key1 or corr[key1][key2]<2: continue;
        key_pair.append([key1,key2])
    def getKey(key_pair):
        global corr
        return int(corr[key_pair[0]][key_pair[1]])
    key_pair=sorted(key_pair,key=getKey,reverse=True)
    print("len", len(key_pair),flush=True)
    return key_pair

####################################################################################
# Functions to run experiments
####################################################################################
def main(thread_index):
    start_time = time.time();

    number_of_keys=1009
    number_of_cores=14
    low_var_adv_tmp = pickle.load(open("data/low_var_adv", 'rb'))
    with open("data/corr_all_key", 'rb') as f:
        corr = pickle.load(f)

    low_var_adv  = [set() for i in range(number_of_keys)]
    for pair in low_var_adv_tmp:
        low_var_adv[pair[1]].add(pair[0])

    adv_key = [set() for i in range(number_of_keys)]
    for key1 in range(number_of_keys):
        if key1%100==0: print(key1,flush=True)
        for key2 in range(number_of_keys):
            if(key2 > key1) and corr[key1][key2]>1:
                with open("data/keys-"+str(key1)+"-"+str(key2)+"/advertiser", 'rb') as f:
                    sharedAdvertisers = pickle.load(f)
                for adv in sharedAdvertisers:
                    if adv not in low_var_adv[key1] and adv not in low_var_adv[key2]:
                        adv_key[key1].add(adv)
                        adv_key[key2].add(adv)

    ijk = 0
    cnt=0
    for key in range(len(adv_key)):
        if len(adv_key[key])<2: continue;
        for adv in adv_key[key]:
            ijk+=1
            if ijk%number_of_cores == thread_index:
                print(key,adv,flush=True)
                generate_pdf_cdf_arrays(key,adv)
    print("Total=",cnt,flush=True)

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
    with open(folder+"pdf_array_virtual_valuation", 'wb') as f:
        pickle.dump(y, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder+"cdf_array_virtual_valuation", 'wb') as f:
        pickle.dump(z, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(time.time()-start,"sec",flush=True)


if __name__ == '__main__' :
    arg=sys.argv
    main(int(arg[1]))
