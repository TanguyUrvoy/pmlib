#!/usr/bin/env python
# coding: utf-8
"""Partial Monitoring Library:
Provides a collection of Finite Partial Monitoring algorithms for experimental studies.
This is REX3, a specialized (adhoc) algorithm for dueling bandits
(see P. Gajane et al. "Non-stochastic utility-based dueling bandits" ICML 2015)
"""
__author__ = "Tanguy Urvoy, Pratik Gajane"
__license__ = "GPL"
__copyright__ = "Orange-labs, France"
__version__ = "1.2"
__email__ = "tanguy.urvoy@orange.com, pratik.gajane@gmail.com"
__date__ = "2017"
__status__ = "Beta"

from policies import BasicPolicy
import numpy as np

from math import log, exp, pow, sqrt
from random import choice
from multiprocessing import cpu_count
from scipy.misc import logsumexp # numerically stable exp sum




# Optimal gamma parameter for finite horizon Rex3 algorithm

def optimal_gamma(pm, horizon):
    K = 1
    while(K*(K+1) < 2*pm.N):
        K += 1
    assert 2*pm.N==K*(K+1)
    assert horizon>0
    
    return min(0.5, sqrt(K*log(K)/horizon))



# Rex3 algorithm

class Rex3(BasicPolicy):
    def __init__(self, pm, gamma):

        self.gamma = gamma
        assert gamma>0
        
        # number of dueling bandit arms
        self.K = 1
        while(self.K*(self.K+1) < 2*pm.N):
            self.K += 1

        print "Number of arms: K=" + str(self.K)

        # check that we have a proper dueling bandit pm
        assert pm.game_type=="dueling"
        assert 2*pm.N == self.K*(self.K+1)
        if pm.M != 2**self.K:
            print "NOTE: non binary rewards"
            

            
        # duels (a,b) to pm action index conversion
        self.duel_to_index_dict = {}
        self.index_to_duel = []
        i=0
        for a in range(self.K):
            for b in range(a, self.K):
                self.index_to_duel.append((a,b))
                self.duel_to_index_dict[(a,b)]=i
                i += 1

        #print "index_to_duel:", self.index_to_duel
        #print
        #print "duel_to_index:", self.duel_to_index_dict
        

        
        # Log Weight 
        self.LW = np.zeros(self.K, dtype=np.float128)
        # Arm distribution
        self.U = np.ones(self.K, dtype=np.float)/self.K
        self.P = np.ones(self.K, dtype=np.float)/self.K
        
        
    def startGame(self): 
        self.LW = np.zeros(self.K, dtype = np.float)
        self.P = np.ones(self.K, dtype=np.float)/self.K

    def choice(self):
        a = np.random.choice(self.K, 1, p = self.P)[0]
        b = np.random.choice(self.K, 1, p = self.P)[0]
        
        # convert (a,b) duel into valid action index
        if(a<=b):
            return self.duel_to_index_dict[(a,b)]
        else:
            return self.duel_to_index_dict[(b,a)]

    
    def getFeedback(self, action, feedback):
        assert action<len(self.index_to_duel)
        assert feedback>=-1 and feedback <= +1
        
        # convert action index into an (a,b) duel
        a,b = self.index_to_duel[action]
        
        np.seterr(over='warn') # rise a warning for float overflow
        
        # Update cum reward estimate
        self.LW[a] += self.gamma/self.K*feedback/(2*self.P[a])
        self.LW[b] -= self.gamma/self.K*feedback/(2**self.P[b])
        
        # Update action selection probability
        # NOTE: We use float128 and logsumexp to postpone the irremediable float saturation problem
        LogZ = logsumexp(self.LW)
        Q = np.exp(self.LW - LogZ)
        Q /= Q.sum()
        self.P = (1.0 - self.gamma) * Q.astype(np.float) +  self.gamma * self.U



    
