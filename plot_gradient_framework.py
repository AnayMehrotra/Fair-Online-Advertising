import helper_gradient_framework as hgf
import pickle, time, os, itertools
import numpy as np
import matplotlib.pyplot as plt


## Hepler to get_rev_ratio
def get_rev_ratio_helper(k1,k2,shift,numAdv,adv,constraints):
    ## Get global keys for errorReporting functions
    global key1;global key2;
    key1=k1;key2=k2

    start_time=time.time()

    numAttr=2

    revenue_constrained   = [ []for p in constraints ]
    revenue_unconstrained   = [ []for p in constraints ]

    for p in range(len(constraints)):
        revenue_constrained_ = [np.array([]),np.array([])]
        revenue_unconstrained_ = [np.array([]),np.array([])]

        for uT in range(numAttr):
            revenue_constrained_[uT]=hgf.get_revenue_array(adv,shift[p][:,uT].reshape(-1,1),uT);
            revenue_unconstrained_[uT]=hgf.get_revenue_array(adv,np.zeros((numAdv,numAttr))[:,uT],uT)
            if revenue_constrained_[uT]==-1 or revenue_unconstrained_[uT]==-1:
                return -1,-1
        revenue_constrained_=np.sum(revenue_constrained_,axis=0)
        revenue_unconstrained_=np.sum(revenue_unconstrained_,axis=0)

        revenue_constrained[p].append(revenue_constrained_);
        revenue_unconstrained[p].append(revenue_unconstrained_);

    print("Time to get stderr: %s seconds " % (time.time() - start_time),flush=True)

    return revenue_constrained,revenue_unconstrained

## Calculate the standard error of the mean of the revenue
def get_rev_ratio():
    start_time = time.time()


    number_of_keys=1009
    constraints = [0,0.1,0.2,0.3,0.4,0.45,0.5]


    ## Results from iter runs of the experiment
    ratio = [ []for p in constraints ]
    result_cons = [ [] for p in constraints ]
    resultUncons = [ [] for p in constraints ]
    result_index = [ ]


    unbalanced_set = get_unbalanced_set(0.2);

    failed_count = 0
    cnt_auc = 0
    bad_keys = [[-1,-1]]#[[10,588],[588,647]]
    for ijk,[key1,key2] in enumerate(unbalanced_set):
        folder="data/keys-"+str(key1)+"-"+str(key2)+"/experiment_gradient_algorithm_Result"
        if not os.path.exists(folder) or [key1,key2] in bad_keys:
            print("Does not exist! "+folder, flush=True)
            failed_count += 1
            continue;
        print("Checking keys:",key1,", ",key2,flush=True)

        # Reference: Structure of data from expriment
        # Result={'numAdv':numAdv,'revenue_constrained':revenue_constrained,
        #         'revenue_unconstrained':revenue_unconstrained,'result_constrained':result_constrained,
        #         'result_unconstrained':result_unconstrained,'result_shift':result_shift,
        #         'constraints':constraints}

        result = pickle.load(open(folder, 'rb') , encoding='latin1')
        shift = result["result_shift"]
        numAdv = result["numAdv"]
        constraints = result["constraints"]
        adv = result["adv"]

        result_cons_,resultUncons_ = get_rev_ratio_helper(key1,key2,shift,numAdv,adv,constraints)

        if result_cons_ == -1 or resultUncons_ == -1:
            failed_count += 1
            continue;

        cnt_auc += 1

        result_index.append(ijk)
        for i in range(len(constraints)):
            result_cons[i].append(result_cons_[i]);
            resultUncons[i].append(resultUncons_[i]);

    pickle.dump(result_cons,open("revenue_constrained","wb"))
    pickle.dump(resultUncons,open("revenue_unconstrained","wb"))

    print("Done running auctions", "successful runs", cnt_auc, "failed runs", failed_count, flush=True)

    ##################################################
    ## Stucture results to store
    ##################################################
    revenue_constrained = pickle.load(open("revenue_constrained","rb"))
    revenue_unconstrained = pickle.load(open("revenue_unconstrained","rb"))

    ratio = [ [] for p in range(len(constraints))]
    ratio_mean = [0.0 for p in range(len(constraints))]
    ratio_err = [0.0 for p in range(len(constraints))]

    number_of_auctions = len(revenue_constrained[0])

    for p in range(len(constraints)):
        tmp = []
        for i in range(10000):
            numerator=0
            denominator=0
            if i%1000==0: print("Number of times auction run",i, flush=True)
            for j in range(number_of_auctions):
                numerator+=max(revenue_constrained[p][j][0][i],0);
                denominator+=max(revenue_unconstrained[p][j][0][i],0);
            tmp.append(numerator/(denominator+1e-12))

        ratio.append(tmp);
        ratio_mean[p]=np.mean(tmp)
        ratio_err[p]=np.std(tmp);

    pickle.dump(ratio_mean,open("revenue_ratio_mean","wb"))
    pickle.dump(ratio_err,open("revenue_ratio_err","wb"))

    ratio_mean = pickle.load(open("revenue_ratio_mean","rb"))
    ratio_err = pickle.load(open("revenue_ratio_err","rb"))

    ##### Get the unfairness of auctions
    print("Time to get stderr: %s seconds " % (time.time() - start_time),flush=True)

    ###################################
    ##Plot unbalanced ratio
    ###################################
    constraints=np.array(constraints)
    ratio_mean=np.array(ratio_mean)
    ratio_err=np.array(ratio_err)

    plt.ylim(0.75, 1.02)
    plt.xlim(-0.01, 0.51)
    plt.errorbar(constraints,ratio_mean,c='orange',yerr=ratio_err,fmt='-',linewidth=4.0)
    plt.xlabel('Fairness constraint $\\ell$',fontsize=20)
    plt.ylabel('Revenue ratio $\\kappa_{\\mathcal{M},\\mathcal{F}}$',fontsize=20)

    plt.savefig('ResultB.eps', format='eps', dpi=500)
    # plt.show()

    return ratio_mean, ratio_err

def get_tv_distance_helper(k1,k2,alpha,numAdv, adv,constraints,split):
    start_time=time.time()

    tv_dist_   = [ [0 for i in range(split)] for p in constraints ]
    print(alpha, flush=True)

    numAttr=2

    for p in range(len(constraints)):
        resCons_ = [np.array([]),np.array([])]
        resUncons_ = [np.array([]),np.array([])]
        for uT in range(numAttr):
            resCons_[uT]=hgf.shiftedMyer(adv,alpha[p][:,uT],uT,stats=0);
            resUncons_[uT]=hgf.shiftedMyer(adv,np.zeros((numAdv,numAttr))[:,uT],uT,stats=0)
        print("Unconstrained",resUncons_, flush=True)
        print("constrained",resCons_, flush=True)
        for i in range(split):
            tv_dist_[p][i] += np.sum(np.abs((resCons_[0][i]+resCons_[1][i])-(resUncons_[0][i]+resUncons_[1][i])))

    print("%s seconds " % (time.time() - start_time),flush=True)

    return tv_dist_

def get_unbalanced_set(val):
    unbalanced = pickle.load(open("implicit_fairness", 'rb'), encoding='latin1')
    print("Done making pairs")
    unbalanced_set = set()
    ijk=-1
    for [k1,k2] in unbalanced['index']:
        ijk+=1
        if unbalanced['unbalance'][ijk]<0.5-val:
            unbalanced_set.add((k1,k2))
    return unbalanced_set

def get_tv_distance():
    start_time = time.time();

    number_of_keys = 1009
    constraints = [0,0.1,0.2,0.3,0.4,0.45,0.5]

    ## Results from iter runs of the experiment
    split = 1
    tv_dist_arr = [ [ [] for i in range(split)] for p in constraints ]
    tv_dist = [ 0.0 for p in constraints ]

    unbalanced_set = get_unbalanced_set(0.2);
    failed_count = 0

    cnt = 0
    cnt_auc = 0

    for ijk,[key1,key2] in enumerate(unbalanced_set):

        folder="data/keys-"+str(key1)+"-"+str(key2)+"/experiment_gradient_algorithm_Result"
        if not os.path.exists(folder):
            print("Does not exist! "+folder)
            failed_count += 1
            continue;

        print("Checking keys:",key1,", ",key2,flush=True)
        result = pickle.load(open(folder, 'rb') , encoding='latin1')
        shift = result["result_shift"]
        numAdv = result["numAdv"]
        constraints = result["constraints"]
        adv = result["adv"]

        tv_dist_ = get_tv_distance_helper(key1,key2,shift,numAdv,adv,constraints,split)

        cnt_auc += 1
        for i,j in itertools.product(range(split),range(len(constraints))):
                tv_dist_arr[j][i].append(tv_dist_[j][0])

    for i in range(len(constraints)): tv_dist_arr[i]=np.array(tv_dist_arr[i][0])
    for i in range(len(constraints)): tv_dist_arr[i]/=2

    pickle.dump(tv_dist_arr, open("tv_dist_arr", 'wb') , protocol=pickle.HIGHEST_PROTOCOL)
    print("Done calculating and saving TV distance", "successful runs", cnt_auc, "failed runs", failed_count)
    print("Time to get tv distance: %s seconds " % (time.time() - start_time),flush=True)

    ###################################
    ##Plot unbalanced ratio
    ###################################
    tv_dist_arr = pickle.load(open("tv_dist_arr", 'rb'))

    mean = np.array([np.mean(tv_dist_arr[i]) for i in range(len(constraints))])
    err = np.array([np.std(tv_dist_arr[i]) for i in range(len(constraints))])

    plt.figure(figsize=(6.6,5))
    plt.errorbar(constraints,mean,yerr=err,c='orange',linewidth=4.0)#fmt='o'
    plt.xlim(0.0, 0.55)
    plt.ylim(0.0, 0.20)
    plt.ylabel('Advertiser Displacement $d_{TV}(\\mathcal{M},\\mathcal{F})$',fontsize=18)
    plt.xlabel('Fairness constraint ($\\ell$)',fontsize=18)
    plt.savefig('ResultC.eps', format='eps', dpi=500)
    # plt.show()

    return mean, err

def plot_unbalance():
    balance = pickle.load(open("implicit_fairness", 'rb'), encoding='latin1')
    balance = np.array(balance['unbalance'])

    cdf = []
    cdf_bins = [0.01*i for i in range(51)]

    for i in cdf_bins: cdf.append(np.sum(balance>i))

    plt.figure(figsize=(8,5))
    plt.plot(cdf_bins,cdf,c='orange')
    plt.xlim(0.0, 0.55)
    plt.ylabel('Number of auctions (key pairs)\n satisfying fairness constraint',fontsize=10)
    plt.xlabel('Fairness constraint ($\ell$)',fontsize=10)
    plt.savefig('Implicit_fairness_of_auctions.eps', format='eps', dpi=500)
    # plt.show()

## Measure the fairness of the mechanisms found
def get_selection_lift():
    start_time = time.time()

    number_of_keys = 1009
    constraints = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]

    ## Results from iter runs of the experiment
    result_sl = [ [] for p in constraints ]

    unbalanced_set = get_unbalanced_set(0.2);

    failed_count = 0
    cnt_auc = 0

    for ijk, [key1,key2] in enumerate(unbalanced_set):

        folder="data/keys-"+str(key1)+"-"+str(key2)+"/experiment_gradient_algorithm_Result"
        if not os.path.exists(folder):
            print("Does not exist! "+folder)
            failed_count += 1
            continue

        print("Checking keys:", key1, ", ", key2, flush=True)

        result = pickle.load(open(folder, 'rb') , encoding='latin1')
        numAdv = result["numAdv"]
        coverage = result["result_constrained"]
        constraints = result["constraints"]

        cnt_auc += 1
        for i in range(len(constraints)):
            for j in range(numAdv):
                x = coverage[i][0][j][0]
                y = coverage[i][0][j][1]
                z = x/(x+y)
                result_sl[i].append(min(z/(1-z),(1-z)/(z+1e-7)));

    pickle.dump(result_sl,open("selection_lift","wb"))

    print("Done running auctions", "successful runs", cnt_auc, "failed runs", failed_count)

    ##################################################
    ## Stucture results to store
    ##################################################
    revenue_sl = pickle.load(open("selection_lift","rb"))

    sl = [ [] for p in range(len(constraints))]
    sl_mean = [0.0 for p in range(len(constraints))]
    sl_err = [0.0 for p in range(len(constraints))]

    for p in range(len(constraints)):
        sl_mean[p]=np.mean(revenue_sl[p])
        sl_err[p]=np.std(revenue_sl[p]);

    print(sl_mean, flush=True)
    print(sl_err, flush=True)

    pickle.dump(sl_mean,open("selection_lift_mean","wb"))
    pickle.dump(sl_err,open("selection_lift_err","wb"))

    sl_mean = pickle.load(open("selection_lift_mean","rb"))
    sl_err = pickle.load(open("selection_lift_err","rb"))

    ##### Get the unfairness of auctions
    print("Time to get selection lift: %s seconds " % (time.time() - start_time), flush=True)

    ###################################
    ##Plot unbalanced ratio
    ###################################
    constraints=np.array(constraints)
    sl_mean=np.array(sl_mean)
    sl_err=np.array(sl_err)

    plt.ylim(0.0, 1.02)
    plt.xlim(-0.01, 0.51)
    plt.errorbar(constraints, sl_mean,c='orange', yerr=sl_err/np.sqrt(3282), fmt='-',linewidth=4.0)
    plt.xlabel('Fairness constraint ($\ell$)',fontsize=20)
    plt.ylabel('Observed Fairness slift($\\mathcal{F}$)',fontsize=20)

    plt.savefig('ResultA.eps', format='eps', dpi=500)
    # plt.show()

    return sl_mean, sl_err

def main():
    #mean1,std1=get_rev_ratio();
    #print("Done with std error calculation", flush=True)
    #print("revenue ratio", flush=True)
    #print(mean1, flush=True)
    #print(std1, flush=True)

    #mean2,std2=get_tv_distance();
    #print("Done with TV norm calculation", flush=True)
    #print("tv distance", flush=True)
    #print(mean2, flush=True)
    #print(std2, flush=True)

    mean3,std3=get_selection_lift();
    print("Done with TV norm calculation", flush=True)
    print("selection lift", flush=True)
    print(mean3, flush=True)
    print(std3, flush=True)

    plot_unbalance()

    print("Results", flush=True)
    #print("revenue ratio", flush=True)
    #print(mean1, flush=True)
    #print(std1, flush=True)
    #print("tv distance", flush=True)
    #print(mean2, flush=True)
    #print(std2, flush=True)
    print("selection lift", flush=True)
    print(mean3, flush=True)
    print(std3, flush=True)

if __name__ == '__main__' :
    main()
