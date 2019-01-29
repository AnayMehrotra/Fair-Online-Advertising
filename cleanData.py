import numpy as np
import csv, pickle, itertools, os
from collections import Counter

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
        for j in range(numKey):
            if corr[i][j]<2 or vis[j] == 1: continue
            cur=cur.union(dfs(j,vis,corr))
        return cur

    # Make blocks
    numKey = 1009
    with open('data/corr_all_key', 'rb') as file:
        corr = pickle.load(file)
    vis = [0 for i in range(numKey)]

    connectedComp = []
    for i in range(numKey):
        if i%100 == 0: print(i)
        if vis[i]==0:
            connectedComp.append(dfs(i,vis,corr))

    connectedComp=sorted(connectedComp,key=len,reverse=True)

    blockCorr = np.zeros((numKey,numKey))
    mp = [0 for i in range(numKey)]
    newIndex = 0

    ## Reindex values
    for component in connectedComp:
        for i in component:
            mp[i] = newIndex
            newIndex += 1

    ## Make new matrix
    for component in connectedComp:
        for r,c in itertools.product(component,component):
                blockCorr[mp[r]][mp[c]] = corr[r][c]

    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    ## Threshold matrix
    blockCorr[blockCorr<2]=0
    blockCorr[blockCorr>=2]=1

    blockCorr[blockCorr==0]=-1
    blockCorr[blockCorr==1]=0
    blockCorr[blockCorr==-1]=1

    print(np.max(blockCorr))

    f, ax = plt.subplots()
    # 'nearest' interpolation - faithful but blocky
    ax.imshow(blockCorr,cmap=cm.Greys_r)
    # ax.imshow(blockCorr,cmap="hot")
    plt.xlabel('Keywords',fontsize=19)
    plt.ylabel('Keywords',fontsize=19)
    plt.xticks([])
    plt.yticks([])
    # plt.title('Keywords sharing at least 2 advertisers')
    plt.savefig('../block_matrix_2.eps', format='eps', dpi=200)
    plt.show()


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
    print("Building arrays.")
    key        = [{"bid":[],"advertiser":[]} for i in range(1009)]
    advertiser = [{"bid":[],"key":[]} for i in range(11009)]

    os.system("mkdir raw_data")

    with open('Webscope_A1/ydata-ysm-advertiser-bids-v1_0.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        print("Reading file.")
        for i,row in enumerate(csv_reader):
            curAdv=int(row[2])
            curKey=int(row[1])
            curBid=float(row[3])
            print(str(curAdv)+", "+str(curKey)+", "+str(curBid))
            key[curKey]["bid"].append(curBid)
            key[curKey]["advertiser"].append(curAdv)
            advertiser[curAdv]["bid"].append(curBid)
            advertiser[curAdv]["key"].append(curKey)

        for i in range(len(advertiser)):
            if(len(advertiser[i]["bid"])==0):
                break;
            with open("raw_data/advertiser-"+str(i), 'wb') as file:
                pickle.dump(advertiser[i], file, protocol=pickle.HIGHEST_PROTOCOL)

        for i in range(len(key)):
            with open("raw_data/key-"+str(i), 'wb') as file:
                pickle.dump(key[i], file, protocol=pickle.HIGHEST_PROTOCOL)


## FUNCTION 2
########################################################
## Clean bids and keys
## Remove advertisers with fewer than 1000 bids
## Remove advertisers with less than 2 keywords
########################################################
def cleanData():
    initialBids = 0
    finalBids = 0
    ## Remove advertisers with less than 1000 bids
    print("Finding bad advertisers")
    toDelete = [set() for i in range(1009)]
    for i in range(10475):
        with open('raw_data/advertiser-'+str(i), 'rb') as fileOpen:
            adv = pickle.load(fileOpen)
            initialBids += len(adv["bid"])
            ## Basic sanity check: remove advertisers bidding less than 2 keywords
            cleanAdv={"bid":[],"key":[]}
            keyMap  = Counter(adv["key"])
            uniqKeys = len(keyMap.keys())
            if i % 100 == 0:
                print(i)
            for k in keyMap.keys():
                if keyMap[k] < 1000:
                    toDelete[k].add(i)
                    uniqKeys -= 1
            if uniqKeys < 2:
                for k in keyMap.keys():
                    toDelete[k].add(i)
                continue
            for j in range(len(adv["bid"])):
                if(keyMap[adv["key"][j]] > 999):
                    cleanAdv["key"].append(adv["key"][j])
                    cleanAdv["bid"].append(adv["bid"][j])
            finalBids += len(cleanAdv["bid"])
            with open('data/advertiser-'+str(i), 'wb') as fileSave:
                pickle.dump(adv, fileSave, protocol=pickle.HIGHEST_PROTOCOL)

    print(100*finalBids/(1.0*initialBids),"% of bids retained.")

    ## Modify bids correspondingly
    print("Modifing keys")
    for i in range(1009):
        print(str(i))
        with open('raw_data/key-'+str(i), 'rb') as fileOpen:
            key = pickle.load(fileOpen)
            cleanKey ={"bid":[],"advertiser":[]}
            for j in range(len(key["bid"])):
                if(key["advertiser"][j] not in toDelete[i]):
                    cleanKey["advertiser"].append(key["advertiser"][j])
                    cleanKey["bid"].append(key["bid"][j])
            with open("data/key-"+str(i), 'wb') as file:
                print("dump clean key")
                pickle.dump(cleanKey, file, protocol=pickle.HIGHEST_PROTOCOL)

## FUNCTION 3
def getPercentageOfBidsRemoved():
    with open("data/lowVarAdv", 'rb') as fileOpen:
        lowVarAdvKeyPair = pickle.load(fileOpen)

    initialBids = 0
    finalBids = 0
        ## Remove advertisers with less than 1000 bids
    print("Finding bad advertisers")
    toDelete = [set() for i in range(1009)]
    for i in range(10475):
        with open('raw_data/advertiser-'+str(i), 'rb') as fileOpen:
            adv = pickle.load(fileOpen)
            initialBids += len(adv["bid"])
            ## Basic sanity check: remove advertisers bidding less than 2 keywords
            cleanAdv={"bid":[],"key":[]}
            keyMap  = Counter(adv["key"])
            uniqKeys = len(keyMap.keys())
            if i % 100 == 0:
                print(i)
            for k in keyMap.keys():
                if keyMap[k] < 1000:
                    toDelete[k].add(i)
                    uniqKeys -= 1
            if uniqKeys < 2:
                for k in keyMap.keys():
                    toDelete[k].add(i)
                continue
            for j in range(len(adv["bid"])):
                if(keyMap[adv["key"][j]] > 999):
                    cleanAdv["key"].append(adv["key"][j])
                    cleanAdv["bid"].append(adv["bid"][j])
                    if [i,adv["key"][j]] not in lowVarAdvKeyPair:
                        finalBids +=1
            with open('data/advertiser-'+str(i), 'wb') as fileSave:
                pickle.dump(adv, fileSave, protocol=pickle.HIGHEST_PROTOCOL)

        print(100*finalBids/(1.0*initialBids),"% of bids retained.")


def main():
    print("getting raw data...")
    getRawData()
    print("got raw data. cleaning data...")
    cleanData()
    print("cleaned data. calculating bids removed...")
    getPercentageOfBidsRemoved()
    print("done.")


if __name__ == '__main__' :
    main()
