import numpy as np
from scipy.stats import truncnorm,uniform
from random import randint
import itertools
import copy, pickle, os, time
import matplotlib.pyplot as plt

numAttr=2;

####################################################################################
# General Functions
####################################################################################
def reset(adv):
    """Reset all Advertisers for new Mechanism"""
    for i in range(len(adv)):
        adv[i].totalWins = 0
        adv[i].result = {}

def collectResult(adv,val,userTy):
    results = [adv[i].getResultG(val,userTy) for i in range(len(adv))]
    return results

####################################################################################
#### Functions for Advertiser class
####################################################################################
def resetResult(self):
    self.totalWins = 0
    self.result = {}

def generateBids(self, type, iter):
    ## pdf: pdf of the given advertiser
    ## inv_cdf: inverse cdf of the given advertiser

    distribution = truncnorm#change the distribution
    r = np.random.rand(int(iter))
    bid = self.inv_cdf[type](r)
    one=np.ones(int(iter))
    virBid = bid-(one-r)/(self.pdf[type](bid)+(one/10000.0))

    return virBid

def bid(self, type, iter):#place a bid given the user

    ## pdf: pdf of the given advertiser
    ## inv_cdf: inverse cdf of the given advertiser

    distribution = truncnorm#change the distribution

    r = np.random.rand(int(iter))
    bid = self.inv_cdf[type](r)
    one=np.ones(int(iter))
    virBid = bid-(one-r)/(self.pdf[type](bid)+(one/10000.0))

    return virBid

def bid2(self, type, iter):#place a bid given the user

    ## pdf: pdf of the given advertiser
    ## inv_cdf: inverse cdf of the given advertiser

    distribution = truncnorm#change the distribution

    # lower,upper=0,self.budget #truncated normal distribution
    # mu, sd = self.mu[type], self.sd[type]
    # a, b = (lower - mu)/sd, (upper - mu)/sd

    # bid = distribution.rvs(a,b,loc=mu,scale=sd,size=iter)

    r = np.random.rand(int(iter))
    bid = self.inv_cdf[type](r)
    one=np.ones(int(iter))
    virBid = bid-(one-r)/(self.pdf[type](bid)+(one/10000.0))

    return bid,virBid

def updateResult(self, userTy):
    #self.cost += price #TODO: Update cost
    self.totalWins += 1
    if userTy in self.result: self.result[userTy] += 1
    else: self.result[userTy] = 1

## Get probability of advertiser winning on a particular user type given he won
def getResultLU(self,val,userTy):
    if userTy not in self.result: return 0
    if self.totalWins == 0: return 0
    else: return self.result[userTy]/self.totalWins

## Get the probability of advertiser winning, given usertype
def getResultG(self,val,userTy):
    if userTy not in self.result: return 0
    if val == 0: return 0
    else:
        return self.result[userTy]/val

####################################################################################
class Advertiser:
    'Common base class for all Advertiser'
    numAdv  = 0
    numAttr = 2

    ### Defined earlier
    resetResult=resetResult
    bid=bid
    bid2=bid2
    getResultLU=getResultLU
    getResultG=getResultG
    updateResult=updateResult

    def __init__(self,cdf,inv_cdf,inv_phi,pdf,range_phi_min):
        global numAttr
        Advertiser.numAdv += 1
        self.cost = 0
        self.index = Advertiser.numAdv
        self.totalWins = 0
        self.result = {}

        self.cdf = []
        self.pdf = []
        self.inv_cdf = []
        self.inv_phi = []
        self.range_phi_min = []

        for i in range(numAttr):
            self.cdf.append(cdf[i])
            self.inv_cdf.append(inv_cdf[i])
            self.inv_phi.append(inv_phi[i])
            self.pdf.append(pdf[i])
            self.range_phi_min.append(range_phi_min[i])
