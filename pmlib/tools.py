#!/usr/bin/env python
# coding: utf-8
"""Partial Monitoring Library:
Provides a collection of Finite Partial Monitoring algorithms for experimental studies.
"""
__author__ = "Tanguy Urvoy, Pratik Gajane"
__copyright__ = "Orange-labs 2017"
__license__ = ""
__version__ = "1.0"
__email__ = "tanguy.urvoy@orange.com, pratik.gajane@gmail.Com"
__date__ = "2017"
__status__ = "Beta"

import numpy as np
import matplotlib.pyplot as plt
from random import choice
from multiprocessing import Pool
from functools import partial

# <h2> PM policies evaluation tools</h2>

# In[35]:

### parallel run of policy on game



def eval_func_tuple(f_args):
    """Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])  

def eval_policy_once(horizon, pm, pol, jobid):
    alg = pol # This is a reference copy but we are using multi-processes (not multi-threads)
    alg.startGame()
    cumRegret = np.zeros(horizon, dtype = np.float)
    
    cumSufferedLoss = 0
    cumAllLosses = np.zeros(pm.N)

    # generate outcomes obliviously
    EnvironmentOutcomes = np.random.choice(pm.M, horizon, p = pm.OutcomeDist)

    for t in range(horizon):
        # policy chooses one action
        action = alg.choice()

        # Environment chooses one outcome
        outcome = EnvironmentOutcomes[t]

        # policy suffers loss and regret
        cumAllLosses += pm.LossMatrix[...,outcome]
        cumSufferedLoss += pm.LossMatrix[action,outcome]
        cumRegret[t] = cumSufferedLoss - min(cumAllLosses)

        # policy gets feedback
        alg.getFeedback(action, pm.FeedbackMatrix[action,outcome])

    return cumRegret
    
def eval_policy_parallel(nbCores, nbReps, horizon, pm, pol):
    print "nbCores:", nbCores, "nbReps:", nbReps, "Horizon:", horizon
    pool = Pool(processes = nbCores)  
    return np.asarray(pool.map(partial(eval_policy_once,horizon,pm,pol), range(nbReps)))

def eval_policy(nbReps, horizon, pm, pol):
    print "(single threaded)", "nbReps:", nbReps, "Horizon:", horizon
    return np.asarray(map(partial(eval_policy_once,horizon,pm,pol), range(nbReps)))



# In[38]:

## Plot stuff

def init_plot(mytitle):
    plt.close()
    plt.grid()
    plt.title(mytitle)
    plt.xlabel("Time")
    plt.ylabel("Regret")


def plot_regret(cumRegrets, mylabel, mycolor = 'black', autoscale = True):
    horizon = cumRegrets.shape[1]
    avgCumRegret = np.mean(cumRegrets, axis=0)
    stdCumRegret = np.std(cumRegrets, axis=0)
    maxCumRegret = np.amax(cumRegrets, axis=0)
    minCumRegret = np.amin(cumRegrets, axis=0)
    tsav = np.asarray(np.linspace(0,horizon-1,200),dtype=int)

    plt.autoscale(autoscale, axis='y')
    plt.fill_between(tsav+1, minCumRegret[tsav], maxCumRegret[tsav], facecolor=mycolor, alpha=0.125)
    for a in range(1,10,2):
        dev = stdCumRegret[tsav] / float(a)
        plt.fill_between(tsav+1,avgCumRegret[tsav]-dev, avgCumRegret[tsav]+dev, facecolor=mycolor, alpha=0.125)
        
    plt.plot(tsav+1, avgCumRegret[tsav], color=mycolor, label=mylabel)

def show_plot():
    plt.legend()
    plt.show()


