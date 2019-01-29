import helper_gradient_framework as hgf
import pickle, time, os, itertools
import numpy as np
import matplotlib.pyplot as plt

def main():
    sys.stdout = open('Plot_experiment_03Jan.txt','w')

    mean1,std1=get_std_error();
    print("Done with std error calculation")

    mean2,std2=get_TV_distance();
    print("Done with TV norm calculation")

    print("Results")
    print("STDERR")
    print(mean1)
    print(std1)
    print("TVNORM")
    print(mean2)
    print(std2)

if __name__ == '__main__' :
    main()

## Hepler to get_std_error
def get_std_error_helper(k1,k2,shift,numAdv,adv,constraints):
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
            revenue_constrained_[uT]=hgf.get_revenue_array(adv,shift[p][:,uT].reshape(-1,1),uT,stats=0);
            revenue_unconstrained_[uT]=hgf.get_revenue_array(adv,np.zeros((numAdv,numAttr))[:,uT],uT,stats=0)
        revenue_constrained_=np.sum(revenue_constrained_,axis=0)
        revenue_unconstrained_=np.sum(revenue_unconstrained_,axis=0)

        revenue_constrained[p].append(revenue_constrained_);
        revenue_unconstrained[p].append(revenue_unconstrained_);

    print("Time to get stderr: %s seconds " % (time.time() - start_time),flush=True)

    return revenue_constrained,revenue_unconstrained

## Calculate the standard error of the mean of the revenue
def get_std_error():
    start_time = time.time()


    number_of_keys=1009
    number_of_cores=45
    constraints = [0,0.1,0.2,0.3,0.4,0.45,0.5]


    ## Results from iter runs of the experiment
    resultRatio = [ [] for p in constraints ]
    resultCons = [ [] for p in constraints ]
    resultUncons = [ [] for p in constraints ]
    result_index = [ ]

    ## Results from iter runs of the experiment
    ratio = [ []for p in constraints ]

    unbalancedSet = get_unbalanced_set(0.2);

    failed_count = 0
    cntAuc = 0
    for ijk,[key1,key2] in enumerate(unbalancedSet):
        folder="data/keys-"+str(key1)+"-"+str(key2)+"/experiment_gradient_algorithm_Result"
        if not os.path.exists(folder):
            print("Does not exist! "+folder)
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

        resultCons_,resultUncons_ = get_std_error_helper(key1,key2,shift,numAdv,adv,constraints)

        cntAuc += 1

        result_index.append(ijk)
        for i in range(len(constraints)):
            resultCons[i].append(resultCons_[i]);
            resultUncons[i].append(resultUncons_[i]);

        pickle.dump(resultCons,open("experiment_gradient_algorithm_result_constrained","wb"))
        pickle.dump(resultUncons,open("experiment_gradient_algorithm_result_unconstrained","wb"))

    print("Done running auctions", "successful runs", cntAuc, "failed runs", failed_count)

    ##################################################
    ## Stucture results to store
    ##################################################
    revenue_constrained = pickle.load(open("experiment_gradient_algorithm_result_constrained","rb"))
    revenue_unconstrained = pickle.load(open("experiment_gradient_algorithm_result_unconstrained","rb"))

    ratio = [ [] for p in range(len(constraints))]
    ratioMean = [0.0 for p in range(len(constraints))]
    ratioErr = [0.0 for p in range(len(constraints))]

    number_of_auctions = len(revenue_constrained[0])

    for p in range(len(constraints)):
        tmp = []
        for i in range(10000):
            numerator=0
            denominator=0
            if i%1000==0: print("Number of times auction run",i)
            for j in range(number_of_auctions):
                numerator+=max(revenue_constrained[p][j][0][i],0);
                denominator+=max(revenue_unconstrained[p][j][0][i],0);
            tmp.append(numerator/(denominator+1e-12))

    ratio.append(tmp);
    ratioMean[p]=np.mean(tmp)
    ratioErr[p]=np.std(tmp);
    pickle.dump(ratioMean,open("ratioMean_experiment_gradient_algorithm","wb"))
    pickle.dump(ratioErr,open("ratioErr_experiment_gradient_algorithm","wb"))

    ratioMean = pickle.load(open("ratioMean_experiment_gradient_algorithm","rb"))
    ratioErr = pickle.load(open("ratioErr_experiment_gradient_algorithm","rb"))

    ##### Get the unfairness of auctions
    print("Time to get stderr: %s seconds " % (time.time() - start_time),flush=True)

    ## Comment to also save plots
    return

    ###################################
    ##Plot unbalanced ratio
    ###################################
    constraints=np.array(constraints)
    constraint=[0,0.1,0.2,0.3,0.4,0.45,0.5]
    ratioMean=np.array(ratioMean)
    ratioErr=np.array(ratioErr)

    plt.ylim(0.75, 1.02)
    plt.xlim(-0.01, 0.51)
    plt.errorbar(constraints,ratioMean,c='orange',yerr=ratioErr,fmt='-',linewidth=4.0)
    plt.xlabel('Fairness constraint $\\ell$',fontsize=20)
    plt.ylabel('Revenue ratio $\\kappa_{\\mathcal{M},\\mathcal{F}}$',fontsize=20)

    plt.savefig('ResultB.eps', format='eps', dpi=500)
    plt.show()

def get_TV_distance_helper(k1,k2,alpha,numAdv, adv,constraints,split):
    start_time=time.time()
    f.iter = 10000

    TVnorm_   = [ [0 for i in range(split)] for p in constraints ]
    print(alpha)

    numAttr=2

    for p in range(len(constraints)):
        resCons_ = [np.array([]),np.array([])]
        resUncons_ = [np.array([]),np.array([])]
        for uT in range(numAttr):
            resCons_[uT]=f.shiftedMyer(adv,alpha[p][:,uT],uT,stats=0);
            resUncons_[uT]=f.shiftedMyer(adv,np.zeros((numAdv,numAttr))[:,uT],uT,stats=0)
        print("Unconstrained",resUncons_)
        print("constrained",resCons_)
        for i in range(split):
            TVnorm_[p][i] += np.sum(np.abs((resCons_[0][i]+resCons_[1][i])-(resUncons_[0][i]+resUncons_[1][i])))

    print("%s seconds " % (time.time() - start_time),flush=True)

    return TVnorm_

def get_unbalanced_set(val):
    unbalanced = pickle.load(open("implicit_fairness", 'rb'), encoding='latin1')
    print("Done making pairs")
    unbalancedSet = set()
    ijk=-1
    for [k1,k2] in unbalanced['index']:
        ijk+=1
        if unbalanced['unbalance'][ijk]<0.5-val:
            unbalancedSet.add((k1,k2))
    return unbalancedSet

def get_TV_distance():
    start_time = time.time();

    number_of_keys=1009
    number_of_cores=45
    constraints=[0,0.1,0.2,0.3,0.4,0.45,0.5]

    ## Results from iter runs of the experiment
    split=1
    TVnormArr = [ [ [] for i in range(split)] for p in constraints ]
    TVnorm = [ 0.0 for p in constraints ]
    resultIndex = [ ]
    unbalancedSet = get_unbalanced_set(0.2);
    failed_count = 0
    cnt=0
    cntAuc = 0
    for ijk,[key1,key2] in enumerate(unbalancedSet):
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
        TVnorm_ = get_TV_distance_helper(key1,key2,shift,numAdv,adv,constraints,split)
        cntAuc += 1
        for i,j in itertools.product(range(split),range(len(constraints))):
                TVnormArr[j][i].append(TVnorm_[j][0])

    for i in range(len(constraints)): TVnormArr[i]=np.array(TVnormArr[i][0])
    for i in range(len(constraints)): TVnormArr[i]/=2

    pickle.dump(TVnormArr, open("TVnormArr_experiment_gradient_algorithm", 'wb') , protocol=pickle.HIGHEST_PROTOCOL)
    print("Done calculating and saving TV norm", "successful runs", cntAuc, "failed runs", failed_count)
    print("Time to get stderr: %s seconds " % (time.time() - start_time),flush=True)

    ###################################
    ##Plot unbalanced ratio
    ###################################
    mean = [np.mean(TVnormArr[i]) for i in range(len(constraints))]
    std = [np.std(TVnormArr[i]) for i in range(len(constraints))]
    TVnormArr = pickle.load(open("TVnormArr_experiment_gradient_algorithm", 'rb'))
    return
    mean = [np.mean(TVnormArr[i]) for i in range(len(constraints))]
    std = [np.std(TVnormArr[i]) for i in range(len(constraints))]
    mean=np.array(mean)
    std=np.array(std)

    plt.figure(figsize=(6.6,5))
    plt.errorbar(constraints,mean,yerr=std,c='orange',linewidth=4.0)#fmt='o'
    plt.xlim(0.0, 0.55)
    plt.ylim(0.0, 0.20)
    plt.ylabel('Advertiser Displacement $d_{TV}(\\mathcal{M},\\mathcal{F})$',fontsize=18)
    plt.xlabel('Fairness constraint ($\\ell$)',fontsize=18)
    plt.savefig('ResultC.eps', format='eps', dpi=500)
    plt.show()

def plotUnbalance():
    balance = pickle.load(open("implicit_fairness", 'rb'), encoding='latin1')
    balance=np.array(balance['unbalance'])

    cdf = []
    cdfbins = [0.01*i for i in range(51)]

    for i in cdfbins: cdf.append(np.sum(balance>i))

    plt.figure(figsize=(8,5))
    plt.plot(cdfbins,cdf,c='orange')
    plt.xlim(0.0, 0.55)
    plt.ylabel('Number of auctions (key pairs)\n satisfying fairness constraint',fontsize=10)
    plt.xlabel('Fairness constraint ($\ell$)',fontsize=10)
    plt.savefig('Implicit_fairness_of_auctions.eps', format='eps', dpi=500)
    plt.show()


## Measure the fairness of the mechanisms found
def getSelectionLift():
    start_time = time.time()

    number_of_keys=1009
    number_of_cores=45
    constraint=[0,0.1,0.2,0.3,0.4,0.45,0.5]
    constraints=[0.5,0.45,0.4,0.30,0.2,0.1,0]


    ## Results from iter runs of the experiment
    resultRatio = [ [] for p in constraints ]
    resultCons = [ [] for p in constraints ]
    resultUncons = [ [] for p in constraints ]
    result_index = [ ]

    ## Results from iter runs of the experiment
    ratio = [ []for p in constraints ]

    unbalancedSet = get_unbalanced_set(0.2);

    failed_count = 0
    cntAuc = 0
    for ijk,[key1,key2] in enumerate(unbalancedSet):
        folder="data/keys-"+str(key1)+"-"+str(key2)+"/experiment_gradient_algorithm_Result"
        if not os.path.exists(folder):
            print("Does not exist! "+folder)
            failed_count += 1
            continue;
        print("Checking keys:",key1,", ",key2,flush=True)

        # Reference: Structure of data from expriment
        # Result={'numAdv':numAdv,'revenue_constrained':revenue_constrained,
        #         'revenue_unconstrained':revenue_unconstrained,'result_constrained':result_constrained,
        #         'result_unconstrained':result_unconstrained,'result_shift':result_shift,
        #         'constraints':constraints}

        result = pickle.load(open(folder, 'rb') , encoding='latin1')
        numAdv = result["numAdv"]
        res = result["result_constrained"]
        constraints = result["constraints"]

        cntAuc += 1
        result_index.append(ijk)
        for i in range(len(constraints)):
            for j in range(numAdv):
                x = res[i][0][j][0]
                y = res[i][0][j][1]
                z = x/(x+y)
                resultCons[i].append(min(z/(1-z),(1-z)/(z+1e-7)));

    pickle.dump(resultCons,open("experiment_gradient_algorithm_selection_lift","wb"))

    print("Done running auctions", "successful runs", cntAuc, "failed runs", failed_count)

    # ##################################################
    # ## Stucture results to store
    # ##################################################
    revenue_selection_lift = pickle.load(open("experiment_gradient_algorithm_selection_lift","rb"))
    # revenue_unconstrained = pickle.load(open("experiment_gradient_algorithm_result_unconstrained","rb"))

    ratio = [ [] for p in range(len(constraints))]
    ratioMean = [0.0 for p in range(len(constraints))]
    ratioErr = [0.0 for p in range(len(constraints))]

    for p in range(len(constraints)):
        ratioMean[p]=np.mean(revenue_selection_lift[p])
        ratioErr[p]=np.std(revenue_selection_lift[p]);

    print(ratioMean)
    print(ratioErr)

    pickle.dump(ratioMean,open("ratioMean_experiment_gradient_algorithm_selection_lift","wb"))
    pickle.dump(ratioErr,open("ratioErr_experiment_gradient_algorithm_selection_lift","wb"))

    mean = pickle.load(open("ratioMean_experiment_gradient_algorithm_selection_lift","rb"))
    std = pickle.load(open("ratioErr_experiment_gradient_algorithm_selection_lift","rb"))

    ##### Get the unfairness of auctions
    print("Time to get stderr: %s seconds " % (time.time() - start_time),flush=True)

    # Comment to save plots
    return

    ###################################
    ##Plot unbalanced ratio
    ###################################ø
    constraints=[0,0.1,0.2,0.3,0.4,0.45,0.5]
    constraints=np.array(constraints)
    mean=np.array(mean)
    std=np.array(std)

    plt.ylim(0.0, 1.02)
    plt.xlim(-0.01, 0.51)
    plt.errorbar(constraints,mean,c='orange',yerr=std/np.sqrt(3282),fmt='-',linewidth=4.0)
    plt.xlabel('Fairness constraint ($\ell$)',fontsize=20)
    plt.ylabel('Observed Fairness slift($\\mathcal{F}$)',fontsize=20)

    plt.savefig('ResultA.eps', format='eps', dpi=500)
    plt.show()
