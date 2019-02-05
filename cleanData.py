import numpy as np
import csv, pickle, itertools, os
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm

########################################################
## Sample data
# 06/15/2002 00:00:00	83	691	.85	0
########################################################

########################################################
## Convert correlation matrix to blocks
########################################################
def plotBlockCorr():
    def dfs(i,vis,corr):
        vis[i]=1
        cur=set([i])
        for j in range(number_of_keys):
            if corr[i][j]<2 or vis[j] == 1: continue
            cur=cur.union(dfs(j,vis,corr))
        return cur

    # Make blocks
    number_of_keys = 1009
    with open('data/corr_all_key', 'rb') as f:
        corr = pickle.load(f)
    vis = [0 for i in range(number_of_keys)]

    connected_comp = []
    for i in range(number_of_keys):
        if i%100 == 0: print(i,flush=True)
        if vis[i]==0:
            connected_comp.append(dfs(i,vis,corr))

    connected_comp=sorted(connected_comp,key=len,reverse=True)

    block_corr = np.zeros((number_of_keys,number_of_keys))
    mp = [0 for i in range(number_of_keys)]
    new_index = 0

    ## Reindex values
    for component in connected_comp:
        for i in component:
            mp[i] = new_index
            new_index += 1

    ## Make new matrix
    for component in connected_comp:
        for r,c in itertools.product(component,component):
                block_corr[mp[r]][mp[c]] = corr[r][c]

    ## Threshold matrix
    block_corr[block_corr<2]=0
    block_corr[block_corr>=2]=1

    block_corr[block_corr==0]=-1
    block_corr[block_corr==1]=0
    block_corr[block_corr==-1]=1

    print(np.max(block_corr),flush=True)

    f, ax = plt.subplots()
    # 'nearest' interpolation - faithful but blocky
    ax.imshow(block_corr,cmap=cm.Greys_r)
    # ax.imshow(block_corr,cmap="hot")
    plt.xlabel('Keywords',fontsize=19)
    plt.ylabel('Keywords',fontsize=19)
    plt.xticks([])
    plt.yticks([])
    # plt.title('Keywords sharing at least 2 advertisers')
    plt.savefig('../block_matrix_2.eps', format='eps', dpi=200)
    plt.show()

## run after distribution fitting
def get_percentage_of_bids_removed():
    with open("data/lowVarAdv", 'rb') as f:
        low_var_adv = pickle.load(f)

    initial_bids = 0
    final_bids = 0

    ## Remove advertisers with less than 1000 bids
    print("Finding bad advertisers",flush=True)
    to_delete = [set() for i in range(1009)]
    for i in range(10475):
        with open('raw_data/advertiser-'+str(i), 'rb') as f:
            adv = pickle.load(f)
            initial_bids += len(adv["bid"])
            ## Basic sanity check: remove advertisers bidding less than 2 keywords
            clean_adv={"bid":[],"key":[]}
            key_map  = Counter(adv["key"])
            uniq_keys = len(key_map.keys())
            if i % 100 == 0:
                print(i,flush=True)
            for k in key_map.keys():
                if key_map[k] < 1000:
                    to_delete[k].add(i)
                    uniq_keys -= 1
            if uniq_keys < 2:
                for k in key_map.keys():
                    to_delete[k].add(i)
                continue
            for j in range(len(adv["bid"])):
                if(key_map[adv["key"][j]] > 999):
                    clean_adv["key"].append(adv["key"][j])
                    clean_adv["bid"].append(adv["bid"][j])
                    if [i,adv["key"][j]] not in low_var_adv:
                        final_bids +=1
            with open('data/advertiser-'+str(i), 'wb') as f:
                pickle.dump(adv, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(100*final_bids/(1.0*initial_bids),"% of bids retained.",flush=True)



## FUNCTION 1
########################################################
## Generate Numpy arrays
## key-i :
##              bids: A list of bids placed by advertisers
##        advertiser: ID of advertiser placing the bid

## advertiser-i :
##              bids: A list of bids placed by advertisers
##               key: ID of keyword on which bid was placed
########################################################
def getRawData():
    print("Building arrays.",flush=True)
    key        = [{"bid":[],"advertiser":[]} for i in range(1009)]
    advertiser = [{"bid":[],"key":[]} for i in range(11009)]

    os.system("mkdir raw_data")

    with open('Webscope_A1/ydata-ysm-advertiser-bids-v1_0.txt') as csv_f:
        csv_reader = csv.reader(csv_f, delimiter='\t')
        print("Reading f.",flush=True)
        for i,row in enumerate(csv_reader):
            cur_adv=int(row[2])
            cur_key=int(row[1])
            cur_bid=float(row[3])
            if i%100000==0: print(str(cur_adv)+", "+str(cur_key)+", "+str(cur_bid),flush=True)
            key[cur_key]["bid"].append(cur_bid)
            key[cur_key]["advertiser"].append(cur_adv)
            advertiser[cur_adv]["bid"].append(cur_bid)
            advertiser[cur_adv]["key"].append(cur_key)

        for i in range(len(advertiser)):
            if(len(advertiser[i]["bid"])==0):
                break;
            with open("raw_data/advertiser-"+str(i), 'wb') as f:
                pickle.dump(advertiser[i], f, protocol=pickle.HIGHEST_PROTOCOL)

        for i in range(len(key)):
            with open("raw_data/key-"+str(i), 'wb') as f:
                pickle.dump(key[i], f, protocol=pickle.HIGHEST_PROTOCOL)


## FUNCTION 2
########################################################
## Clean bids and keys
## Remove advertisers with fewer than 1000 bids
## Remove advertisers with less than 2 keywords
########################################################
def cleanData():
    os.system("mkdir data")

    initial_bids = 0
    final_bids = 0
    ## Remove advertisers with less than 1000 bids
    print("Finding bad advertisers",flush=True)
    to_delete = [set() for i in range(1009)]
    for i in range(10475):
        with open('raw_data/advertiser-'+str(i), 'rb') as f:
            adv = pickle.load(f)
            initial_bids += len(adv["bid"])
            ## Basic sanity check: remove advertisers bidding less than 2 keywords
            clean_adv={"bid":[],"key":[]}
            key_map  = Counter(adv["key"])
            uniq_keys = len(key_map.keys())
            if i % 100 == 0:
                print(i,flush=True)
            for k in key_map.keys():
                if key_map[k] < 1000:
                    to_delete[k].add(i)
                    uniq_keys -= 1
            if uniq_keys < 2:
                for k in key_map.keys():
                    to_delete[k].add(i)
                continue
            for j in range(len(adv["bid"])):
                if(key_map[adv["key"][j]] > 999):
                    clean_adv["key"].append(adv["key"][j])
                    clean_adv["bid"].append(adv["bid"][j])
            final_bids += len(clean_adv["bid"])

            with open('data/advertiser-'+str(i), 'wb') as f:
                pickle.dump(adv, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(100*final_bids/(1.0*initial_bids),"% of bids retained.",flush=True)

    ## Modify bids correspondingly
    print("Modifing keys",flush=True)
    for i in range(1009):
        if i%100==0: print(str(i),flush=True)
        with open('raw_data/key-'+str(i), 'rb') as f:
            key = pickle.load(f)
            clean_key ={"bid":[],"advertiser":[]}
            for j in range(len(key["bid"])):
                if(key["advertiser"][j] not in to_delete[i]):
                    clean_key["advertiser"].append(key["advertiser"][j])
                    clean_key["bid"].append(key["bid"][j])
            with open("data/key-"+str(i), 'wb') as f:
                if i%100==0: print("dump clean key",flush=True)
                pickle.dump(clean_key, f, protocol=pickle.HIGHEST_PROTOCOL)

## FUNCTION 3
def get_correlation_and_bids():
    number_of_keys = 1009

    ## keys: Details of key
    keys = []
    for i in range(number_of_keys):
        if i%100==0: print(i,flush=True)
        with open('data/key-'+str(i), 'rb') as f:
            tmp = pickle.load(f)
            keys.append(tmp)

    ## List of (Set of keywords for each Advertiser)
    ## good_adv[i] = Set of top keywords for advertiser i
    good_adv = []
    ## List of (Count of bids by some advertiser for some keyword)
    ## cnt_bid_adv[i][j] = Count of bids by advertiser j for keyword i
    cnt_bid_adv = []
    tot=0
    for kkk,k in enumerate(keys):
        if kkk%100==0: print(kkk,flush=True)
        ## cnt_adv[i][0]: (count of of bids by advertiser i on current key (k) )
        ## cnt_adv[i][1]: i
        cnt_adv=[[0,i] for i in range(12000)]
        ## calculate values of cnt_adv
        for a in k["advertiser"]:
            cnt_adv[a][0]+=1
        ## cnt_adv[i]: touple(count of of bids by i-th largest advertiser on current key (k) , index of advertiser)
        cnt_adv.sort(reverse=True)

        ## cur_adv: Set of advertisers for current key
        cur_adv=set()
        ## tmp[i] number of bids by advertiser i on current keyword
        ## Stores data for 20 largest advertisers
        tmp={}
        for i in range(len(cnt_adv)):
            ## Only take advertisers who bid more that 1000 times.
            if cnt_adv[i][0]>1000:
                tot+=cnt_adv[i][0]
                tmp[cnt_adv[i][1]]=cnt_adv[i][0]
                cur_adv.add(cnt_adv[i][1])

        cnt_bid_adv.append(tmp)
        good_adv.append(cur_adv)

    with open("data/cnt_bid_adv", 'wb') as f:
        pickle.dump(cnt_bid_adv, f, protocol=pickle.HIGHEST_PROTOCOL)

    ## Corr: correlation matrix of bids
    ## corr[i][j]: number of advertiser shared between key i and j, only advertisers among top 20 for each
    corr = np.zeros((number_of_keys,number_of_keys))
    for i,j in itertools.product(range(number_of_keys),range(number_of_keys)):
            corr[i][j]+=len(good_adv[i].intersection(good_adv[j]));

    with open("data/"+"corr_all_key", 'wb') as f:
        pickle.dump(corr, f, protocol=pickle.HIGHEST_PROTOCOL)

    to_delete = set()
    ## Remove bids which don't share any keyword
    for i in range(number_of_keys):
        Del=True
        for j in range(number_of_keys):
            if(j<=i):continue
            if corr[i][j]>1:
                Del=False
        if Del: to_delete.add(i);


    ## Tot: total bids placed on top 50 keywords
    ## cnt: total bids places on suitable pair of top 50 keywords
    tot=number_of_keys*number_of_keys;
    cnt=0;
    for i,j in itertools.product(range(number_of_keys),range(number_of_keys)):
        if j>i and corr[i][j]>=2:
            cnt+=1
            folder="data/keys-"+str(i)+"-"+str(j)
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(folder+"/advertiser", 'wb') as f:
                pickle.dump(good_adv[i].intersection(good_adv[j]), f, protocol=pickle.HIGHEST_PROTOCOL)

    tot=number_of_keys*number_of_keys;
    print("Fraction of suitable pairs: "+str(cnt/(tot*1.0)),flush=True)
    a=np.zeros((number_of_keys,number_of_keys))
    a[corr>=2]=1;
    print("Fraction of suitable pairs: "+str(a.mean()/2),flush=True)


def main():
    print("getting raw data...",flush=True)
    getRawData()
    print("got raw data. cleaning data...",flush=True)
    cleanData()
    print("cleaned data. finding correlation and counting bids...",flush=True)
    get_correlation_and_bids()
    print("done.",flush=True)


if __name__ == '__main__' :
    main()
