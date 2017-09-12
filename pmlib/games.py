#!/usr/bin/env python
# coding: utf-8
"""Partial Monitoring Library:
Provides a collection of Finite Partial Monitoring algorithms for experimental studies.
"""
__author__ = "Tanguy Urvoy"
__copyright__ = "Orange-labs, France"
__license__ = "GPL"
__version__ = "1.2"
__email__ = "tanguy.urvoy@orange.com"
__date__ = "2017"
__status__ = "Beta"

import numpy as np
import matplotlib.pyplot as plt
from math import log, exp, pow
from scipy.optimize import linprog
from random import choice
from multiprocessing import Pool

## A list of PM games examples
benchmark_games = []
benchmark_names = []



## Generic Partial Monitoring(PM) instance structure

class PMGame(object):
    __slots__ = ['N', 'M', 'OutcomeDist', 'LossMatrix', 'FeedbackMatrix', 'FeedbackMatrix_symb', 'Actions_dict', 'Outcomes_dict',
                 'title', 'game_type']
    def __init__(self, N, M, title =""):
        self.N = N # Number of learner actions
        self.M = M # Number of environment outcomes
        self.OutcomeDist = np.ones(M, dtype=np.float)/M # outcome distribution (for stochastic PM)
        self.LossMatrix = np.ones(shape=(N,M), dtype=np.float) # Loss MAtrix
        self.FeedbackMatrix = np.empty(shape=(N,M), dtype=np.float) # Feedback (numeric form)
        self.FeedbackMatrix_symb = np.empty(shape=(N,M), dtype=object) # Feedback (symbolic)        
        self.Actions_dict = { a : "{0}".format(a) for a in range(self.N)} # Actions semantic
        self.Outcomes_dict = { a : "{0}".format(a) for a in range(self.M)} # Outcomes semantic
        self.title = title
        self.game_type = "generic"

    # print game matrices and plot outcomes distribution
    # if plot is true : output a distribution plot
    # if nice is true use panda for nice jupyter output
    def dump(self, plot = False, nice = False, show_numeric=False):
        if(len(self.title)>0):        
            print "***** " + self.title + " *****"
        print "Actions: N=" +str(self.N),"Outcomes: M=" + str(self.M)
        if not nice:
            print "Actions semantic:"
            print self.Actions_dict
            print "Outcomes semantic:"
            print self.Outcomes_dict
        print
        print "Loss Matrix (with actions as row indices and outcomes as column indices):"
        if nice:
            import pandas
            from IPython.display import display,HTML
            df_loss = pandas.DataFrame(self.LossMatrix, columns=self.Outcomes_dict.values(), index=self.Actions_dict.values())
            display(df_loss)
        else:
            print self.LossMatrix
        print
        print "Feedback Matrix (symbolic form):"
        if nice:
            df_feedback = pandas.DataFrame(self.FeedbackMatrix_symb, columns=self.Outcomes_dict.values(), index=self.Actions_dict.values())
            display(df_feedback)
        else:
            print self.FeedbackMatrix_symb
        if show_numeric:
            print "Feedback Matrix (numeric form):"
            if nice:
                df_feedback = pandas.DataFrame(self.FeedbackMatrix, columns=self.Outcomes_dict.values(), index=self.Actions_dict.values())
                display(df_feedback)
            else:
                print self.FeedbackMatrix
        
        print
        print "Outcomes distribution (for stochastic games):"
        for a in range(self.M):
            print "P(" + self.Outcomes_dict[a] + ")=" + str(self.OutcomeDist[a]),

        if(plot==True):
            if(len(self.title)>0):
                plt.title("Outcomes probabilities for " + self.title)
            else:
                plt.title("Outcomes probabilities")
            plt.axis('equal')
            plt.pie(self.OutcomeDist,labels=self.Outcomes_dict.values())
            plt.show()


# Outcome encoding helper for binary reward vectors of dimension K (Binary Bandits helper)
# NOTE: low-bits encode high arm indices
def is_set(x, n, K):
    return x & 2**(K-n-1) != 0 

def arm_reward(x, n, K):
    if is_set(x,n,K):
        return 1.
    else:
        return 0.


#########################
### Bernoulli Bandit  ###
#########################

# Generate a PM instance for Bernoulli Bandit 
def BernoulliBandit(Arms):
    K = len(Arms) # number of arms
    Arms = np.array(Arms, dtype=float)
    pm = PMGame(K,2**K, str(K)+"-armed bandit") # PM Game with K actions and 2**K outcomes
    pm.game_type = "bandit"
    
    
    pm.Outcomes_dict = { a : "{0:b}".format(a).zfill(int(log(pm.M,2))) for a in range(pm.M)}
    pm.Actions_dict= { a : "arm {0}".format(a) for a in range(K) }
  
    ## 1 - Each outcome is a binary reward vector of dimension K encoded as an integer
    for x in range(pm.M):
        px = 1.
        for a in range(K):
            if is_set(x,a,K):
                px *= Arms[a]
            else:
                px *= 1. - Arms[a]
        pm.OutcomeDist[x] = px

    ## 2 - Loss and Feedback matrices
    for a in range(K):
        for x in range(pm.M):
            pm.LossMatrix[a,x] = 1.0 - arm_reward(x,a,K)
            pm.FeedbackMatrix[a,x] = arm_reward(x,a,K)
            if arm_reward(x,a,K):
                pm.FeedbackMatrix_symb[a,x] = 'win '
            else:
                pm.FeedbackMatrix_symb[a,x] = 'loss'
    return pm

benchmark_games.append(BernoulliBandit([0.9,0.5,0.1]))
benchmark_names.append("Easy Bandit")
benchmark_games.append(BernoulliBandit([0.6,0.5,0.5,0.5]))
benchmark_names.append("Hard Bandit")


###################################
### Dynamic pricing PM problem  ###
###################################

# Generate a PM instance for Dynamic pricing problem
def DynamicPricingPM(PriceDist, storage_cost):
    M = len(PriceDist)
    pm = PMGame(M,M, str(M)+"-levels dynamic pricing") # It's a PM Game with both M actions and outcomes
    pm.game_type = "dynamic pricing"
    
    pm.OutcomeDist = np.array(PriceDist, dtype=float)
    pm.OutcomeDist /= pm.OutcomeDist.sum()
    
    for price in range(M):
        pm.Actions_dict[price] = str(price)+"$"
        for budget in range(M):
            if budget >= price: # sell!
                pm.LossMatrix[price,budget] = budget - price
                pm.FeedbackMatrix[price,budget] = 1
                pm.FeedbackMatrix_symb[price,budget] = 'sold'
            else: # no sell!
                pm.LossMatrix[price,budget] = storage_cost
                pm.FeedbackMatrix[price,budget] = 0
                pm.FeedbackMatrix_symb[price,budget] = 'not-sold'
    for budget in range(M):
        pm.Outcomes_dict[budget] = str(budget)+"$"
    
    return pm


benchmark_games.append(DynamicPricingPM([0.1,0.1,0.7,0.1], 2.))
benchmark_names.append("Four levels easy Dynamic Pricing (c=2)")
benchmark_games.append(DynamicPricingPM([0.3,0.1,0.1,0.1,0.1,0.3], 2.))
benchmark_names.append("Five levels hard Dynamic Pricing (c=2)")

    

################################
### Bartok's toy PM problem  ###
################################


# Generate a PM instance for G. Bartok's thesis toy PM example
# This is a simple locally observable game
# See [G. Bartok 'The Role of Information in Online Learning', 2012] on page 88
def BartokPM(Dist):
    pm = PMGame(3,3,"Bartok game") # It's a PM Game with 3 actions and 3 outcomes
    pm.game_type = "Bartok"
    
    assert len(Dist) == 3
    pm.OutcomeDist = np.array(Dist, dtype=float)
    
    pm.LossMatrix = np.array(
     [[1, 1, 0],
     [0, 1, 1],
     [1, 0, 1]], dtype = float)

    pm.FeedbackMatrix = np.array(
        [['1', '0', '0'],
        ['0', '1', '0'],
         ['0', '0', '1']], dtype=np.float)

    pm.FeedbackMatrix_symb = np.array(
        [['a', 'b', 'b'],
         ['b', 'a', 'b'],
         ['b', 'b', 'a']], dtype=object)
        
    return pm

benchmark_games.append(BartokPM([1/3.,1/3.,1/3.]))
benchmark_names.append("G. Bartok's thesis game")


#################################
### Apple tasting PM problem  ###
#################################

# Generate a PM instance for the Apple Tasting problem
# This is a simple destructive test example
# See [N.Cesa-Bianchi & G. Lugosi 'PRediction Learning and Games', 2006] chapter 6
def AppleTasting(Dist):
    pm = PMGame(2,2,"Apple tasting game") # It's a PM Game with 2 actions and 2 outcomes
    pm.game_type = "apple tasting"
    assert len(Dist) == 2
    pm.OutcomeDist = np.array(Dist, dtype=float)
    
    pm.LossMatrix = np.array(
     [[1, 0],
      [0, 1]], dtype = float)

    pm.FeedbackMatrix = np.array(
            [[0, 0],
             [+1, -1]], dtype=np.float)

    pm.FeedbackMatrix_symb = np.array(
            [['blind', 'blind'],
             ['rotten', 'good']], dtype=object)

    pm.Actions_dict = { 0:'sell apple', 1:'taste apple'}
    pm.Outcomes_dict = { 0:'rotten', 1:'good'}
    
    return pm

benchmark_games.append(AppleTasting([0.05,0.95]))
benchmark_names.append("Apple tasting (organic food)")
benchmark_games.append(AppleTasting([0.50,0.50]))
benchmark_names.append("Apple tasting (supermarket)")
                

#################################################
### Full-feedback online learning PM problem  ###
#################################################

def FullFeedback(Dist = [0.1,0.6,0.1,0.2]):
    N = len(Dist)
    pm = PMGame(N, N, "Full-information (horse race)")
    pm.game_type ="full-feedback"
    pm.OutcomeDist = np.array(Dist, dtype=np.float)
    pm.OutcomeDist /= pm.OutcomeDist.sum()

    pm.LossMatrix = 1 - np.diag(np.ones(N))
    pm.FeedbackMatrix = np.vstack([np.array(range(N)) for i in range(N)])
    pm.FeedbackMatrix_symb = pm.FeedbackMatrix
    
    pm.Actions_dict = { i:"bet on horse " + str(i) for i in range(N)}
    pm.Outcomes_dict = { j:str(j) for j in range(N) }
    return pm



benchmark_games.append(FullFeedback())
benchmark_names.append("Horse race")
    

####################################
### A simple intractable problem ###
####################################


# Generate a trivially intractable PM game
def Intractable(Dist):
    pm = PMGame(2,2,"Intractable") # It's a PM Game with 2 actions and 2 outcomes
    pm.game_type = "intractable"    
    assert len(Dist) == 2
    pm.OutcomeDist = np.array(Dist, dtype=float)
    pm.OutcomeDist /= pm.OutcomeDist.sum()
        
    pm.LossMatrix = np.array(
     [[1, 0],
      [0, 1]], dtype = np.float)

    pm.FeedbackMatrix = np.array(
        [[1, 1],
         [2, 2]], dtype=np.float)

    pm.FeedbackMatrix_symb = np.array(
        [['maybe', 'maybe'],
         ['who-knows', 'who-knows']], dtype=object)

    pm.Actions_dict = { 0:'ask', 1:'not-ask'}
    pm.Outcomes_dict = { 0:'no', 1:'yes'}
    
    return pm


benchmark_games.append(Intractable([0.75,0.25]))
benchmark_names.append("Intractable")

######################################
### Label efficient online learning ##
######################################


# Generate a PM instance for the Label-Efficient prediciton problem
# where the learner has to pay to get feedback
# See [N.Cesa-Bianchi & G. Lugosi 'PRediction Learning and Games', 2006] chapter 6
def LabelEfficientPrediction(Dist):
    pm = PMGame(3,2,"Label-efficient prediction") # It's a PM Game with 2 actions and 2 outcomes
    pm.game_type = "label-efficient"    
    assert len(Dist) == 2
    pm.OutcomeDist = np.array(Dist, dtype=float)
    pm.OutcomeDist /= pm.OutcomeDist.sum()
    
    pm.LossMatrix = np.array(
     [[1, 1],
      [0, 1],
      [2, 0]], dtype = np.float)

    pm.FeedbackMatrix = np.array(
        [[-1, +1],
         [0, 0],
         [0, 0]], dtype=np.float)

    pm.FeedbackMatrix_symb = np.array(
        [['ham', 'spam'],
         ['blind', 'blind'],
         ['blind', 'blind']], dtype=object)

    pm.Actions_dict = { 0:'ask user', 1:'transfer email', 2:'drop email'}
    pm.Outcomes_dict = { 0:'ham', 1:'spam'}
    
    return pm

#FIXME: there was a bug around
benchmark_games.append(LabelEfficientPrediction([0.75,0.25]))
benchmark_names.append("Label efficient prediction")
                

###################################
### Dueling Bandit PM matrices  ###
###################################

# Generate a PM instance for Binary utility-based Dueling Bandit
# as in [Gajane and Urvoy EWRL 2015]
#
def BinaryUtilityDuelingBanditPM(Arms):
    Arms = np.array(Arms, dtype = np.float)
    
    # 1 - start as a Bernoulli Bandit
    pm = BernoulliBandit(Arms)
    pm.game_type = "dueling"    
    
    # 2 - Actions spcae is different:
    K = Arms.shape[0]
    pm.N=K*(K+1)/2
    pm.title = str(K) + "-armed utility-based dueling bandit"

    # 3 - build PM matrices

    pm.LossMatrix = np.zeros(shape=(pm.N,pm.M))
    pm.FeedbackMatrix = np.empty(shape=(pm.N,pm.M), dtype=np.float)
    pm.FeedbackMatrix_symb = np.empty(shape=(pm.N,pm.M), dtype=object)    

    id = 0
    for a in range(0,K):
        for b in range(a,K):
            pm.Actions_dict[id] = "({0},{1})".format(a,b)
            for x in range(0,pm.M):
                pm.LossMatrix[id,x] = 1. - (arm_reward(x,a,K) + arm_reward(x,b,K)) / 2.

                if arm_reward(x,a,K) > arm_reward(x,b,K):
                    pm.FeedbackMatrix_symb[id,x] = 'win ' # a win againt b
                elif arm_reward(x,a,K) < arm_reward(x,b,K):
                    pm.FeedbackMatrix_symb[id,x] = 'loss' # a lose against b
                else:
                    pm.FeedbackMatrix_symb[id,x] = 'tie ' # tie match
                
                if arm_reward(x,a,K) > arm_reward(x,b,K):
                    pm.FeedbackMatrix[id,x] = +1 # a win againt b
                elif arm_reward(x,a,K) < arm_reward(x,b,K):
                    pm.FeedbackMatrix[id,x] = -1 # a lose against b
                else:
                    pm.FeedbackMatrix[id,x] = 0 # tie match

            id += 1
            
    
    
    return pm

benchmark_games.append(BinaryUtilityDuelingBanditPM([0.9,0.5,0.1]))
benchmark_names.append("Easy Dueling Bandit")
benchmark_games.append(BinaryUtilityDuelingBanditPM([0.6,0.5,0.5,0.5]))
benchmark_names.append("Hard Dueling Bandit")







if __name__ == "__main__":

    
    for i in range(len(benchmark_games)):
        print
        print
        game = benchmark_games[i]
        game.dump(plot=False, nice=True) # set plot=True to plot the outcome distributions







