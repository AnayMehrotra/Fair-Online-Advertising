import gradientFramework as gf
import helper_gradient_framework as hgf
import numpy as np
import copy, time, pickle, itertools, sys
import matplotlib.pyplot as plt

#### Set up and global
with open("data/corr_all_key", 'rb') as fileOpen:
    corr = pickle.load(fileOpen)

numAttr=2; #numAttr: Number of attributes
numAdv=-1; #Number of advertisers


def runExperiment(thread_index):
    #index if the
    start_time = time.time();
    sys.stdout = open('<Specify file name to record the run>'+'_Thread'+str(thread_index)+'_<Specify date>.txt','w')

    number_of_keys=1009
    number_of_cores=45
    constraint=[0,0.1,0.2,0.3,0.4,0.45,0.5]

    ## Results from iter runs of the experiment
    ratio = [ []for p in constraint ]
    keyPair = hgf.getKeyPair(number_of_keys,number_of_cores,corr)

    unbalanced = pickle.load(open("implicit_fairness", 'rb'), encoding='latin1')
    unbalancedSet = set()

    for ijk,[k1,k2] in enumerate(unbalanced['index']):
        if unbalanced['unbalance'][ijk]<0.3:
            unbalancedSet.add((k1,k2))

    for ijk,[key1,key2] in enumerate(unbalancedSet):
        if ijk % number_of_cores == thread_index:
            print("Running keys:",key1,", ",key2,flush=True)
            _ = runAuction(key1,key2,constraint,thread_index)

    print("Time to run experiment: %s seconds " % (time.time() - start_time),flush=True)

def runAuction(k1,k2,constraints,thread,stats=0):
    global numAttr;global numAdv;global key1; global key2
    key1=k1;key2=k2

    start_time = time.time();

    ## Get advertisers
    gf.adv, gf.y, gf.z= hgf.getAdv(k1,k2)
    gf.removeAdv()
    gf.numAdv=len(gf.adv)
    adv=copy.deepcopy(gf.adv)
    numAdv=len(gf.adv)

    if(gf.numAdv<2):
        hgf.reportError("Only "+str(gf.numAdv)+ " advertiser",key1,key2);
        return
    print("numAdv:",gf.numAdv,flush=True)

    gf.samples=60000
    gf.min_x = -30; gf.max_x =  100
    gf.x=np.linspace(gf.min_x,gf.max_x,gf.samples)

    revenue_unconstrained =[[]for p in constraints]
    revenue_constrained   =[[]for p in constraints]
    result_unconstrained =[[]for p in constraints]
    result_constrained   =[[] for p in constraints]
    result_shift  =[[]for p in constraints]

    initial_shift = np.zeros((numAttr,len(adv)))

    for i in range(len(constraints)):
        ## Initialize
        consLu = np.zeros((numAttr,numAdv))
        for a in range(numAttr): consLu[a] = constraints[i]
        gf.m = np.zeros((numAttr,numAdv))
        gf.set_constraint(consLu)

        shift=np.zeros((numAttr,len(adv)))
        res=np.zeros((numAdv,numAttr));

        if(stats): print(">>Myerson...");
        for attr in range(numAttr):res[:,attr]=gf.coverage_total(shift[attr].reshape(-1,1),attr);
        if(stats): printResult(res,shift);

        revenue_unconstrained[i].append(gf.revenue_total(shift))
        result_unconstrained[i].append(res)

        shift = np.zeros((numAttr,numAdv))
        if i != 0:
            shift=copy.deepcopy(initial_shift)

        if(stats): print(">>Optimizing...");start_time = time.time();
        shift, loss_self, loss_algorithm2 = gf.GDRevenue(shift);
        if(stats): printResult(res,shift);

        rev=gf.revenue_total(np.zeros((numAttr,numAdv)));

        ## Uncomment to generate convergence plots
        # index=[i for i in range(len(loss_self))]
        # plt.semilogy(index,(rev-loss_self)/rev)
        # plt.ylabel('Revenue')
        # plt.xlabel('Iterations')
        # plt.title('Algorithm 1, keys 2-5')
        # plt.show();
        #
        # lelen = max([len(tmp) for tmp in loss_algorithm2])
        # plt.ylabel('Loss $L$')
        # plt.xlabel('Iterations')
        # plt.title('Algorithm 2, keys 2-5')
        # legend=[]
        # for j,loss in enumerate(loss_algorithm2):
        #     index=[i for i in range(min(500,len(loss)))]
        #     plt.semilogy(index,loss[:500])
        #     legend.append("Run:"+str(j))
        # plt.legend(legend)
        # plt.show();

        for attr in range(numAttr):res[:,attr]=gf.coverage_total(shift[attr].reshape(-1,1),attr);
        revenue_constrained[i].append(gf.revenue_total(shift))
        result_constrained[i].append(res)
        result_shift[i].append(shift)

        initial_shift = copy.deepcopy(shift)

        if stats:
            print("Result: ",res.tolist())
            print("shift: ",shift.tolist())
            print("revenue",i,"= ",revenue_constrained[i]);

    ##################################################
    ## Stucture results to store
    ##################################################
    result_constrained=np.array(result_constrained)
    result_unconstrained=np.array(result_unconstrained)
    tmprevenue_constrained=revenue_constrained
    revenue_constrained=np.array(revenue_constrained).T[0]

    result_shift = np.array(result_shift)

    tmprevenue_constrained=revenue_unconstrained
    revenue_unconstrained=np.array(revenue_unconstrained).T[0]

    ##################################################
    ## Save auction result
    ##################################################
    Result={'numAdv':numAdv,'adv':adv,'revenue_constrained':revenue_constrained,'revenue_unconstrained':revenue_unconstrained,'result_constrained':result_constrained,'result_unconstrained':result_unconstrained,'result_shift':result_shift,'constraints':constraints}
    folder="data/keys-"+str(key1)+"-"+str(key2)+"/"
    with open(folder+"experiment_gradient_algorithm_Result", 'wb') as file:
        pickle.dump(Result, file, protocol=pickle.HIGHEST_PROTOCOL)

    ## Print shift for strongest fairness constraint
    print(result_constrained[0])
    print(result_constrained[-1])
    print("Shift for p="+str(constraints[-1]), result_shift[-1].tolist(),flush=True)
    print("Results for p="+str(constraints[0]), hgf.toRelative(result_constrained[0][0],numAdv).tolist(),flush=True)
    print("Results for p="+str(constraints[-1]), hgf.toRelative(result_constrained[-1][0],numAdv).tolist(),flush=True)
    print("--- %s seconds ---" % (time.time() - start_time),flush=True)

    ratio = [ revenue_constrained[i]/(revenue_unconstrained[i]+1e-7) for i in range(len(constraints))]
    return ratio


def get_unbalanced_keys():
    number_of_keys=1009
    number_of_cores=45
    constraint=[0.0]
    sys.stdout = open('get_unbalanced_keys'+'_'+'<Specify date>'+'.txt','w')

    ## Results from iter runs of the experiment
    ratio = [ []for p in constraint ]
    keyPair = hgf.getKeyPair(number_of_keys,number_of_cores,corr)

    unbalance = {"unbalance":[], "index":[]}
    for ijk,[key1,key2] in enumerate(keyPair):
        start_time = time.time();
        print("Running keys:",key1,", ",key2,flush=True)

        ## Get advertisers
        gf.adv, gf.y, gf.z= hgf.getAdv(key1,key2)
        if gf.y == -1: continue
        gf.removeAdv()
        gf.numAdv=len(gf.adv)
        adv=copy.deepcopy(gf.adv)
        numAdv=len(gf.adv)

        if(gf.numAdv<2):
            print("ERROR Only "+str(gf.numAdv)+ " advertiser");
            continue;
        print("numAdv:",gf.numAdv,flush=True)

        gf.samples=60000
        gf.min_x = -30; gf.max_x =  100
        gf.x=np.linspace(gf.min_x,gf.max_x,gf.samples)

        shift=np.zeros((numAttr,len(adv)))
        res=np.zeros((numAdv,numAttr));

        for attr in range(numAttr):res[:,attr]=gf.coverage_total(shift[attr].reshape(-1,1),attr);
        relative_result = hgf.toRelative(res,numAdv)
        print(relative_result)
        unbalance_ = 100
        for a,i in itertools.product(range(numAttr),range(numAdv)):
            unbalance_ = min(unbalance_, relative_result[i][a])
        unbalance["unbalance"].append(unbalance_)
        unbalance["index"].append([key1,key2])
        print("Unbalance",unbalance_)
        print(time.time()-start_time,"seconds")

        with open("implicit_fairness", 'wb') as file:
            pickle.dump(unbalance,file)
