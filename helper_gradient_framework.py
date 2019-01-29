import pickle
from advertiser import *

####################################################################################
# Helper Functions
####################################################################################
def reportError(msg,key1,key2):
    os.system("echo \""+str(key1)+"-"+str(key2)+": "+msg+"\">> errorsExperiment")

def toRelative(res,numAdv):
    ## Convert "probability of advertiser winning, given usertype" to
    ## "probability of advertiser winning on a particular user type given he won"
    resRel=copy.deepcopy(res)
    for i in range(numAdv):
        tmp=resRel[i][0]+resRel[i][1];
        resRel[i][0]/=tmp;resRel[i][1]/=tmp
    resRel=np.around(resRel, decimals=10)
    return resRel

####################################################################################
# Functions to gather data
####################################################################################
corr=0 # correlation between different keys
def getKeyPair(number_of_keys,number_of_cores,corr_local):
    global corr
    corr = corr_local

    keyPair=[]
    for key1 in range(number_of_keys):
        for key2 in range(number_of_keys):
            if key2<=key1 or corr[key1][key2]<2: continue;
            keyPair.append([key1,key2])

    def getKey(keyPair):
        global corr
        return int(corr[keyPair[0]][keyPair[1]])

    keyPair=sorted(keyPair,key=getKey,reverse=True)
    print("numer of key pairs found", len(keyPair))
    return keyPair

def getAdv(key1,key2):
    folder="data/keys-"+str(key1)+"-"+str(key2)+"/"
    ## Maximum number of advertisers
    sharedAdvertisers = pickle.load(open(folder+"/advertiser", 'rb'))
    lowVarAdvKeyPair = pickle.load(open("data/lowVarAdv", 'rb'))
    lowVarAdv  = set()
    for pair in lowVarAdvKeyPair:
        if pair[1]==key1 or pair[1]==key2:
            lowVarAdv.add(pair[0])

    sharedAdvertisers = sharedAdvertisers.difference(lowVarAdv)
    numAdv=len(sharedAdvertisers);
    if(len(sharedAdvertisers)<2):
        print("numAdv:",numAdv,flush=True)
        reportError("Only "+str(numAdv)+ " advertiser",key1,key2);
        return -1,-1,-1

    ## Folder to load pdf and cdf
    folder="data/keys-"+str(key1)+"-"+str(key2)+"/"

    ##################################################
    ## Get distributions of advertisers
    ##################################################
    cdf=[[] for i in range(numAdv)];
    pdf=[[] for i in range(numAdv)];
    inv_cdf=[[] for i in range(numAdv)];
    inv_phi=[[] for i in range(numAdv)];
    range_phi_min=[[] for i in range(numAdv)]

    pdf_arr = [[] for a in range(numAttr)]
    cdf_arr = [[] for a in range(numAttr)]

    i=0
    for adv in sharedAdvertisers:
        if i >= numAdv: break
        folder="data/keys-"+str(key1)+"-adv"+str(adv)+"/"
        tmpcdf = pickle.load(open(folder+"cdf", 'rb'))
        tmpinv_cdf = pickle.load(open(folder+"inv_cdf", 'rb'))
        tmppdf = pickle.load(open(folder+"pdf", 'rb'))
        tmpinv_phi = pickle.load(open(folder+"inv_phi", 'rb'))
        tmprange_phi_min = pickle.load(open(folder+"range_phi_min", 'rb'))
        cdf[i].append(tmpcdf[0])
        pdf[i].append(tmppdf[0])
        inv_cdf[i].append(tmpinv_cdf[0])
        inv_phi[i].append(tmpinv_phi[0])
        range_phi_min[i].append(tmprange_phi_min[0])

        pdf_arr[0].append(pickle.load(open(folder+"pdf_array_virtual_valuation", 'rb')))
        cdf_arr[0].append(pickle.load(open(folder+"cdf_array_virtual_valuation", 'rb')))
        i+=1
    i=0
    for adv in sharedAdvertisers:
        if i >= numAdv: break
        folder="data/keys-"+str(key2)+"-adv"+str(adv)+"/"
        tmpcdf = pickle.load(open(folder+"cdf", 'rb'))
        tmpinv_cdf = pickle.load(open(folder+"inv_cdf", 'rb'))
        tmppdf = pickle.load(open(folder+"pdf", 'rb'))
        tmpinv_phi = pickle.load(open(folder+"inv_phi", 'rb'))
        tmprange_phi_min = pickle.load(open(folder+"range_phi_min", 'rb'))
        cdf[i].append(tmpcdf[0])
        pdf[i].append(tmppdf[0])
        inv_cdf[i].append(tmpinv_cdf[0])
        inv_phi[i].append(tmpinv_phi[0])
        range_phi_min[i].append(tmprange_phi_min[0])

        pdf_arr[1].append(pickle.load(open(folder+"pdf_array_virtual_valuation", 'rb')))
        cdf_arr[1].append(pickle.load(open(folder+"cdf_array_virtual_valuation", 'rb')))
        i+=1

    adv = []
    for i in range(numAdv):
        adv.append(Advertiser(cdf[i],inv_cdf[i],inv_phi[i],pdf[i],range_phi_min[i]))

    return adv,pdf_arr,cdf_arr

def getPdfCdf(a,i):
    iter=100*samples
    bid = np.linspace(min_x,max_x,iter)

    cdf_list = adv[i].cdf[a](bid)
    one=np.ones(int(iter))
    virBid = bid-(one-cdf_list)/(adv[i].pdf[a](bid)+(one/100000.0))

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


def get_revenue_array(adv,shift,uT):
    iter = 10000;

    if len(adv) != len(shift):
        print("Error! len(adv) != len(shift)"+str(len(adv))+"!="+str(len(shift)))

    numAdv=len(adv)
    bids = [ [] for i in range(numAdv)]
    virB = [ [] for i in range(numAdv)]

    # Getting bids
    for i in range(numAdv):
        bids[i],virB[i]=adv[i].bid2(uT,10+int(iter))
    bids=np.array(bids);virB=np.array(virB);

    # Calculate the winner
    abc = virB+shift.reshape((-1,1))
    winner = np.argmax(abc,axis=0)
    runUp = np.array([-1 for i in range(int(iter))])
    for j in range(int(iter)):
        run=-1;se=-100;
        for i in range(numAdv):
            if i == winner[j]: continue;
            if se<abc[i][j]:
                se=abc[i][j];
                runUp[j]=i;

    # Calculating payment
    pay=[]
    query = [[] for i in range(numAdv)]
    when  = [[] for i in range(numAdv)]
    who   = []
    revenueArray=[];

    j=0;i=0;
    for it in range(int(iter)):
        j=runUp[it];i=winner[it]
        value=0
        value=virB[j][it]-shift[i]+shift[j]
        value=max(value,adv[i].range_phi_min[uT])
        if type(value) != int and type(value) != float and type(value)!= np.float64:
            value=value[0]
        ## Bluk queries are much faster
        query[i].append(value)
        when[i].append(it)
        who.append(i)

    ## Get answers to bulk queries
    ans = [ adv[i].inv_phi[uT](query[i]) for i in range(numAdv)]
    cnt = [0 for i in range(numAdv)]
    for it in range(int(iter)):
        j=runUp[it];i=who[it]
        tmp=ans[i][cnt[i]]
        tmp=min(tmp,bids[j][it])
        revenueArray.append(tmp)
        cnt[i]+=1

    return revenueArray
