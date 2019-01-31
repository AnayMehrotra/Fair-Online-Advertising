#!/usr/bin/python
from random import randint
import numpy as np
import copy, time
import matplotlib.pyplot as plt
from scipy.stats import norm,expon,truncnorm,uniform
from scipy import integrate
from numpy import linalg

from advertiser import *
from helper_gradient_framework import *

# Array of advertisers
adv = ["Add","something","useful"]
samples=60000
# Minimum and maximum virtual valuation
min_x = -30; max_x =  100
x=np.linspace(min_x,max_x,samples)

####################################################################################
# Gradient framework
####################################################################################

## Gradient oracle

def shift_index(s): return int(samples * s/(max_x-min_x));
def get_shift(shift,j):
    max_s=max(shift)[0];
    min_s=min(shift)[0];

    a=shift_index(max_s-shift[j])
    b=shift_index(min_s-shift[j])-1
    if a-b == shift_index(max_s-shift[0])-shift_index(min_s-shift[0])+1-1:
        a+=1
    elif a-b == shift_index(max_s-shift[0])-shift_index(min_s-shift[0])+1+1:
        if b != -1: b+=1
        else: a-= 1
    elif a-b != shift_index(max_s-shift[0])-shift_index(min_s-shift[0])+1:
        a+=shift_index(max_s-shift[0])-shift_index(min_s-shift[0])+1+b-a
        print("Error! #1",shift_index(max_s-shift[0])-shift_index(min_s-shift[0])+1+b-a,flush=True)

    if a-b != shift_index(max_s-shift[0])-shift_index(min_s-shift[0])+1:
        print("Error! #2",a,b,shift_index(max_s-shift[0]),shift_index(min_s-shift[0])+1,flush=True)
    return a,b
# Calculates q_{shift,attr}
def coverage_advertiser(shift,i,attr):
    max_s=max(shift)[0];
    min_s=min(shift)[0];
    a,b=get_shift(shift,i)
    tmp = y[attr][i][a:b]
    for j in range(len(shift)):
        if j != i:
            a,b=get_shift(shift,j)
            tmp=tmp*z[attr][j][a:b]
    q=integrate.simps(tmp, dx=((max_x-min_x)/samples))
    return q
# Partial gradient \frac{q_{i, attr}}{\alpha_{j, attr}}
def partial_gradient_coverage(shift, i, j,attr):
    max_s=max(shift)[0];
    min_s=min(shift)[0];
    def g(i):
        a,b=get_shift(shift,i)
        ans = np.zeros(len(x[a:b]))
        for j in range(len(shift)):
            if j != i: ans += -f(i,j);
        return ans
    def f(i,j):
        a,b=get_shift(shift,i)
        tmp = -y[attr][i][a:b]
        a,b=get_shift(shift,j)
        tmp = tmp* y[attr][j][a:b]

        for k in range(len(shift)):
            if k != i and k != j:
                a,b=get_shift(shift,k)
                tmp=tmp*z[attr][k][a:b]
        return tmp
    if i==j: q=integrate.simps(g(i), dx=((max_x-min_x)/samples))
    else: q=integrate.simps(f(i,j), dx=((max_x-min_x)/samples))
    return q
# revenue for user type attr and shift=shift
def revenue_attribute(shift,attr):
    max_s=max(shift)[0];
    min_s=min(shift)[0];
    def f(i):
        a,b=get_shift(shift,i)
        tmp=x[a:b]*y[attr][i][a:b]
        for j in range(len(shift)):
            if j != i:
                a,b=get_shift(shift,j)
                tmp=tmp*z[attr][j][a:b]
        return tmp
    for i in range(len(shift)):
        if i == 0: arr=f(i)
        else: arr+=f(i)

    revenue=integrate.simps(arr, dx=((max_x-min_x)/samples))
    return revenue
# Total revenue across all user types
def revenue_total(shift):
    rev=0;
    for a in range(numAttr): rev+=revenue_attribute(shift[a].reshape(-1,1),a);
    return rev
# calculates the partial gradient, \frac{rev}{alpha_{i,attr}}
def partial_gradient_revenue_shift(shift,i,attr):

    max_s=max(shift)[0];
    min_s=min(shift)[0];
    def f(j):
        a,b=get_shift(shift,i);tmp=x[a:b]
        a,b=get_shift(shift,i);tmp=tmp*y[attr][i][a:b]
        a,b=get_shift(shift,j);tmp=tmp*y[attr][j][a:b]
        for k in range(len(shift)):
            if k != i and k != j:
                a,b=get_shift(shift,k)
                tmp=tmp*z[attr][k][a:b]
        return tmp
    def g(j):
        a,b=get_shift(shift,j);tmp=x[a:b]
        a,b=get_shift(shift,i);tmp=tmp*y[attr][i][a:b]
        a,b=get_shift(shift,j);tmp=tmp*y[attr][j][a:b]
        for k in range(len(shift)):
            if k != i and k != j:
                a,b=get_shift(shift,k)
                tmp=tmp*z[attr][k][a:b]
        return tmp
    a,b=get_shift(shift,0)
    arr = np.zeros(len(x[a:b]))
    for j in range(len(shift)):
        if j != i : arr+=f(j)
    for j in range(len(shift)):
        if j != i : arr-=g(j)
    revenue=integrate.simps(arr, dx=((max_x-min_x)/samples))
    return revenue
# calculates the Jacobian, after fixing one advertiser to 0 for each user type
def jacobian_coverage_transpose(shift,attr):
    max_s=max(shift)[0];
    min_s=min(shift)[0];
    jacoTrans = np.zeros((len(shift),len(shift)));
    for i in range(len(shift)):
        for j in range(len(shift)):
            if i==j: continue
            jacoTrans[j][i] = partial_gradient_coverage(shift,i,j,attr);
    for i in range(len(shift)):
        sum=0
        for j in range(len(shift)):
            sum+=jacoTrans[i][j]
        jacoTrans[i][i]=-sum
    for i in range(len(shift)):
        jacoTrans[-1][i]=0
        jacoTrans[i][-1]=0
    return jacoTrans
# calculates the gradient \nabla q_{i,attr}
def gradient_coverage(shift,i,attr):
    nabla = []
    for j in range(len(shift)-1): nabla.append(partial_gradient_coverage(shift,i,j,attr))
    nabla.append(0)
    return np.array(nabla)
# calculates the gradient \nabla rev_{shift}
def gradient_revenue_shift(shift,attr):
    nabla = []
    for j in range(len(shift)-1): nabla.append(partial_gradient_revenue_shift(shift,j,attr))
    nabla.append(0)
    return np.array(nabla)
# [Answer of the gradient oracle] calculates the gradient \nabla rev w.r.t. coverage q
def gradient_revenue_coverage(shift,attr):

    jacoTrans = jacobian_coverage_transpose(shift,attr);
    jacoTransInv = linalg.pinv(jacoTrans)

    return np.matmul(jacoTransInv,gradient_revenue_shift(shift,attr).reshape(-1,1)).T[0]
# calculates q_{attr} = [q_{1,attr},q_{2,attr},\dots,q_{n,attr}]
def coverage_total(shift,attr):
    return np.array([coverage_advertiser(shift,i,attr) for i in range(len(shift))])
# calculates \nabla \mathcal{L}(\alpha)
def get_grad_loss(adv,target,shift,attr):
    coeff = np.array([-2* (target[i] - coverage_advertiser(shift,i,attr)) for i in range(len(shift))])
    nabla  = []
    for j in range(len(shift)-1):
        tmp=0
        for i in range(len(shift)-1):
            tmp += partial_gradient_coverage(shift,i,j,attr)
        nabla.append(tmp)
    nabla.append(0)
    nabla = np.array(nabla)
    return nabla * coeff


## Projection oracle (when m=2, i.e., two user types)
# Matrix of slopes defining constraints for all i\in[n] and j\in[m]
m = np.zeros((2,100))
# Find the slope corresponding to the constraint
def set_constraint(l):
    global m
    m[0] = (l[1])/(1-l[1]+0.00000001);
    m[1] = (1-l[0])/(l[0]+0.0000001)

def projection_helper(x,y,m):
    r=(m * y + x)/(m ** 2 + 1);
    return r, m * r
def violates(x,y,m):
    return y > m * x
# Project onto the set of lower bound constraints
def projection_constraint(q):

    ans=copy.deepcopy(q)

    for i in range(numAdv):
        if violates(ans[0][i],-ans[1][i],-m[0][i]):
            ans[0][i],ans[1][i] = projection_helper(ans[0][i],ans[1][i],m[0][i])
        elif violates(ans[0][i],ans[1][i],m[1][i]):
            ans[0][i],ans[1][i] = projection_helper(ans[0][i],ans[1][i],m[1][i])
    return ans
# Project onto the simplex of single item constraints
def projection_simplex(x, mask=None):

    if mask is not None:
        mask = np.asarray(mask)
        xsorted = np.sort(x[~mask])[::-1]
        # remaining entries need to sum up to 1 - sum x[mask]
        sum_ = 1.0 - np.sum(x[mask])
    else:
        xsorted = np.sort(x)[::-1]
        # entries need to sum up to 1 (unit simplex)
        sum_ = 1.0
    lambda_a = (np.cumsum(xsorted) - sum_) / np.arange(1.0, len(xsorted)+1.0)
    for i in range(len(lambda_a)-1):
        if lambda_a[i] >= xsorted[i+1]:
            astar = i
            break
    else:
        astar = -1
    p = np.maximum(x-lambda_a[astar],  0)
    if mask is not None:
        p[mask] = x[mask]
    return p
# Projection on the fair polytope, $\mathcal{Q}$ using Dykstra's projection algorithm
def projection(x):
    p=np.zeros((numAttr,numAdv))
    q=np.zeros((numAttr,numAdv))
    y=np.zeros((numAttr,numAdv))
    x_prev=np.ones((numAttr,numAdv))
    i = 0
    while linalg.norm(x-x_prev) > 1e-6:
        y[0]=projection_simplex(x[0]+p[0]);
        y[1]=projection_simplex(x[1]+p[1]);

        i += 1
        p=x+p-y
        x_prev=copy.deepcopy(x)
        x=projection_constraint(y+q)
        q=y+q-x
    return x

## Gradient descent algorithm
def GDCoverage(adv, target, shift,attr):
    ## Track loss
    loss=[]

    ## Hyper-parameters
    gamma = 6000 * len(adv)
    eps = 1 * 1e-5

    ## Initialize
    q=coverage_total(shift,attr)
    prev_err = linalg.norm(q-target) ** 2
    loss.append(prev_err)

    i=0 ## Total iterations so far
    j=0 ## Number of iterations when error decreased
    while prev_err > eps:
        i+=1
        j+=1 # j=0 if error increased in this iteration
        if j % 40 * int(i/100+1) == 39 * int(i/100+1) and prev_err > 1e-4:
            ## Increase learning rate if current error is large and learning rate is performing well
            gamma *= 4
        if i > 1000 and prev_err < 1e-4:
            break
        if i > 2000: ## Stop if error is not reducing
            print("GDCoverage quit early!",flush=True)
            break

        ## Calculate gradient
        grad=get_grad_loss(adv,target,shift,attr)

        ## Helpful print statements
        # if i%500==499:
        #     i+=0
        #     print("       error in last iteration:", prev_err, flush=True)
        #     print("       learning rate",gamma, flush=True)
        #     print("       current shift:",shift.reshape(-1,1).tolist(), flush=True)
        #     print("       gradient:",grad.tolist(), flush=True)
        #     print("       current coverage",q.tolist(), flush=True)

        ## Update learning rate when gradient is large
        gamma = min(gamma, 3/(max(abs(grad))+0.00001));

        ## Gradient Update
        shift_tmp = shift - gamma * grad.reshape(-1,1)
        q = coverage_total(shift_tmp,attr)

        ## Reduce learning rate if loss increases
        if gamma > 1 and (prev_err - linalg.norm(q-target) ** 2)/prev_err < -0.1:
            gamma *= 0.5
            q = coverage_total(shift,attr)
            j=0  ## Error increased
        else:
            shift=copy.deepcopy(shift_tmp)

        ## Track loss
        prev_err = linalg.norm(q-target) ** 2
        loss.append(prev_err)

    print("        Final coverage from GDCoverage(): ",q, flush=True)
    print("        Final coverage error from GDCoverage(): ",prev_err, flush=True)

    return shift, loss
def GDRevenue(shift):
    # Track loss
    loss_self=[]
    loss_algorithm2 = []

    ## Hyper-parameters (learning rate and stopping error)
    gamma = np.array([0.05 for a in range(numAttr)]);
    eps_value = 1e-5
    eps_sequence = 1e-2
    step_size=1 ## Stop when gradient step is small

    ## Initialize coverage as the projection of optimal on fair polytope
    q = np.zeros((numAttr,numAdv));
    for a in range(numAttr): q[a]=coverage_total(shift[a].reshape(-1,1),a);
    q = projection(q)

    for a in range(numAttr):
        shift_tmp, _= GDCoverage(adv,q[a],shift[a].reshape(-1,1),a)
        shift[a]=shift_tmp.reshape((numAdv));
    for a in range(numAttr): q[a]=coverage_total(shift[a].reshape(-1,1),a);
    print("Initial coverage",q, flush=True)

    ## Initial revenue
    rev=revenue_total(shift);
    rev_prev=0.5*rev

    i=0
    print_period = 10 # Show updates every {#1} iterations
    revenue_decreased = False # True if revenue decreased

    while step_size > eps_sequence or revenue_decreased == True or i < 100:
        i += 1
        if i > 500: break;
        revenue_decreased = False

        grad = [[]for a in range(numAttr)]
        for a in range(numAttr): grad[a] = gradient_revenue_coverage(shift[a].reshape(-1,1),a)

        if i % print_period == 0:
            print("Iteration",i,", revenue in last iteration ",rev_prev, flush=True)
            # print("     step_size in last iteration", step_size, flush=True)
            print("    gradient", grad, flush=True)
            # print("     shift", shift, flush=True)
            # print("    rev", rev, flush=True)
            print("    learning rate",gamma, flush=True)
            #print("    current coverage ", q.tolist(), flush=True)

        qprev= copy.deepcopy(q)
        shift_prev = copy.deepcopy(shift)
        q_tmp = np.zeros((numAttr,numAdv));

        ## Update learning rate when gradient is large
        for a in range(numAttr): gamma[a] = min(gamma[a], 0.1/(max(grad[a])+0.00001));

        ## Gradient step
        done = np.zeros(numAttr)
        for a in range(numAttr):
            if done[a]==True: continue
            q_tmp[a] = q[a] + gamma[a] * grad[a]

        ## Project on the fair polytope
        q_tmp = projection(q_tmp)

        ## Update next iteration
        q=q_tmp

        #if i % print_period == 0: print("     next coverage ", q.tolist, flush=True)

        ## Get shift from Algorithm2 (GDCoverage)
        for a in range(numAttr):
            shift_tmp,rum_loss_algorithm2 = GDCoverage(adv,q[a],shift[a].reshape(-1,1),a)
            shift[a]=shift_tmp.reshape((numAdv));

        ## Optionally use iterative binary search
        # res,shift = oracle2(adv,q,shift);

        ## Update step due to error from Algorithm2
        for a in range(numAttr): q[a]=coverage_total(shift[a].reshape(-1,1),a);

        ## Track loss
        loss_algorithm2.append(rum_loss_algorithm2)
        rev_prev=copy.deepcopy(rev)
        rev=revenue_total(shift);
        loss_self.append(rev)

        q_prev_step = np.zeros((numAttr,numAdv));

        ## Calculate step size using a for learning rate of 0.05
        for a in range(numAttr):
            q_prev_step[a] = qprev[a] + gamma[a] * grad[a]
        q_prev_step = projection(q_prev_step)
        step_size = linalg.norm(projection(qprev)-q_prev_step)

        if i % print_period == 0:
            print("Relative improvement (revenue_current-revenue_previous)/revenue_previous",(rev - rev_prev)/rev_prev, flush=True)

        ## Reduce learning rate if loss increases
        ## also undo current gradient step
        if (rev - rev_prev)/rev_prev < -eps_value:
            revenue_decreased = True
            gamma *= 0.8
            shift=copy.deepcopy(shift_prev)
            q=copy.deepcopy(qprev)
            rev=rev_prev
            if max(gamma) < 1e-8:
                break
        elif (rev - rev_prev)/rev_prev > -eps_value and (rev - rev_prev)/rev_prev < eps_value:
            break

    return shift,loss_self,loss_algorithm2

## Remove advertisers who win less than x% of the time
def removeAdv():
    while len(adv) > 0:
        numAdv=len(adv)
        res=np.zeros((numAdv,numAttr));
        for attr in range(numAttr): res[:,attr]=coverage_total(np.zeros((numAttr,len(adv)))[attr].reshape(-1,1),attr);

        # print(res)
        cnt=0
        delete=[]
        for i in range(numAdv):
            ## Thresholding set lower threshold on winning probability of advertisers
            if res[i][0]<0.05 or res[i][1]<0.05:
                delete.append(i)
        delete.reverse()
        for d in delete:
                del y[0][d]
                del y[1][d]
                del z[0][d]
                del z[1][d]
                del adv[d]
                cnt+=1
        numAdv-=cnt;
        if cnt == 0: break


    numAdv=len(adv)
    res=np.zeros((numAdv,numAttr));
    for attr in range(numAttr): res[:,attr]=coverage_total(np.zeros((numAttr,len(adv)))[attr].reshape(-1,1),attr);

    print("Result after removing advertisers",flush=True)
    print(res,flush=True)
