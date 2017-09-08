#!/usr/bin/env python
# coding: utf-8
"""Partial Monitoring Library:
Provides a collection of Finite Partial Monitoring algorithms for experimental studies.
"""
__author__ = "Tanguy Urvoy, Pratik Gajane"
__copyright__ = "Orange-labs 2017"
__license__ = "LGPL"
__version__ = "1.0"
__email__ = "tanguy.urvoy@orange.com, pratik.gajane@gmail.Com"
__date__ = "2017"
__status__ = "Beta"

import numpy as np

from math import log, exp, pow, sqrt
from scipy.optimize import linprog
from random import choice
from multiprocessing import cpu_count
import games
from policies import BasicPolicy
import tools
from scipy.misc import logsumexp # numerically stable exp sum

# FeedExp3 core algorithm
# As in [Piccolboni Schindelhauer "Discrete Prediction Games with Arbitrary Feedback and Loss" 2000]
# and [Bianchi et al. 2006 "Regret minimization under partial monitoring"]
# Use action x outome input matrices
# Require a numeric feedback matrix F where there exists a link-matrix K s.t. L=KF
# If no eta,gamma parameters are provided, we uses anytime adaptive parameters

class BasicFeedexp3(BasicPolicy):
    def __init__(self, pm, eta = 0, gamma = 0):
        # Prepare matrices for basic FeedExp3
        self.N, self.M = pm.N, pm.M
        self.LossMatrix = pm.LossMatrix
        try:
            self.FeedbackMatrix = pm.FeedbackMatrix.astype(np.float, casting='unsafe')
        except ValueError as err:
            raise ValueError("Invalid Feedback Matrix: " + str(err))
        
        # sanity check
        assert self.N,self.M == self.LossMatrix.shape
        assert self.N,self.M == self.FeedbackMatrix.shape
        
        loss_rank = np.linalg.matrix_rank(self.LossMatrix)
        feedback_rank = np.linalg.matrix_rank(np.concatenate((self.LossMatrix,self.FeedbackMatrix),axis=0))
        if(feedback_rank < loss_rank):
            print "WARNING: Feedbak rank (", feedback_rank, ") is lower than loss rank (", loss_rank, ")"
                            
        # Compute Feedexp3 Link matrix
        self.LinkMatrix = np.linalg.lstsq(self.FeedbackMatrix.transpose(), self.LossMatrix.transpose())[0].transpose()
        
        # Set constant values
        # As specified in [Bianchi et al. "Regret minimization under partial monitoring", 2006]
        # and [N.Cesa-Bianchi & G. Lugosi 'PRediction Learning and Games', 2006] chapter 6
        self.kstar = max(1.,np.fabs(self.LinkMatrix).max())
        C = pow(self.kstar * sqrt(exp(1.)-2.), 2./3.)
        self.eta0 = pow(log(self.N)/self.N, 2./3.) / C
        self.gamma0 = C * pow(self.N*self.N*log(self.N), 1./3.)
        self.eta = eta
        self.gamma = gamma
            
        self.U = np.ones(self.N, dtype = np.float)/self.N # uniform distribution
        self.L = np.zeros(self.N, dtype = np.float)
        self.P = np.ones(self.N, dtype = np.float)/self.N
        self.t = 1

        # FIXME: add a proper verbose/trace system
        # print "NxM={0}x{1}".format(self.N, self.M)
        # print "Feedexp3-LossMatrix="
        # print self.LossMatrix
        # print "Feedexp3-FeedbackMatrix="
        # print self.FeedbackMatrix
        # print
        print "kstar=", self.kstar, "eta0=", self.eta0, "gamma0=", self.gamma0
        print "{0} x {0} Link matrix".format(self.LinkMatrix.shape[0])
        #print self.LinkMatrix
        err = np.linalg.norm(self.LinkMatrix.dot(self.FeedbackMatrix) - self.LossMatrix)
        print "Loss estimation error:", err
        
        
    def startGame(self): 
        self.L = np.zeros(self.N, dtype = np.float)
        self.P = np.ones(self.N, dtype= np.float)/self.N
        self.t = 1
        

    def choice(self):
        return np.random.choice(self.N, 1, p = self.P)[0]

    
    def getFeedback(self, action, feedback):
        assert action<self.N
        np.seterr(over='warn') # rise a warning for float overflow
        
        # Update parameters
        self.t += 1
        
        ## If parameters are set to zero, take the dynamic theoretical values 
        ## as specified in [Bianchi et al. 2006 "Regret minimization under partial monitoring"]
        if self.eta==0:
            eta = self.eta0 / pow(self.t, 2./3.)
        else:
            eta = self.eta
        if self.gamma==0:
            gamma = np.fmin(1.,self.gamma0 / pow(self.t, 1./3.))
        else:
            gamma = self.gamma

        # Update by-action cumulative loss estimate
        self.L += self.LinkMatrix[:,action] * feedback / self.P[action]

        # Update action selection probability
        # NOTE: We use float128 and logsumexp to postpone the irremediable float saturation problem
        LogZ = logsumexp(-eta*self.L)
        Q = np.exp(-eta*self.L - LogZ)
        Q /= Q.sum()
        self.P = (1.0 - gamma) * Q.astype(np.float) +  gamma * self.U


## Original [Piccolboni Schindelhauer "Discrete Prediction Games with Arbitrary Feedback and Loss" 2000]
## fixed-known-horizon settings
def optimal_Feedexp3_parameters(pm, horizon):
    eta = pow(log(pm.N), 1./2.) / pow(horizon, 1./2.)
    gamma = np.fmin(1.,pow(pm.N, 1./2.) * pow(log(pm.N),1./4.) / pow(horizon, 1./4.))
    return eta, gamma


if __name__ == "__main__":
    Arms = np.array([0.75,0.5,0.5])
    horizon = 10000
    nbReps = 32
    nbCores = cpu_count() / 2
    pm_game = games.BernoulliBandit(Arms)
    
    print "Bernoulli Bandit arms parameters:", Arms
    print
    pm_game.dump()
    tools.show_plot()
    print
    

    tools.init_plot("FeedExp3 Partial Monitoring Regret on a " + str(len(Arms)) + "-armed Bandit")

    ## Random policy

    print "== Random Baseline =="
    baseline = BasicPolicy(pm_game)
    cumRegrets = tools.eval_policy_parallel(nbCores, nbReps, horizon, pm_game, baseline)
    tools.plot_regret(cumRegrets, mylabel= "Random", mycolor = 'red')

    ## Basic FeedExp3 policy

    print "== BasicFeedExp3 (fixed horizon) =="
    eta, gamma = optimal_Feedexp3_parameters(pm_game, horizon)
    bfx3 = BasicFeedexp3(pm_game, eta , gamma)
    cumRegrets = tools.eval_policy_parallel(nbCores, nbReps, horizon, pm_game, bfx3)
    tools.plot_regret(cumRegrets, mylabel= "Basic FeedExp3 (fixed horizon)", mycolor = 'black')

    ## Basic FeedExp3 policy

    print "== BasicFeedExp3 (anytime) =="
    bfx3 = BasicFeedexp3(pm_game, 0, 0)
    cumRegrets = tools.eval_policy_parallel(nbCores, nbReps, horizon, pm_game, bfx3)
    tools.plot_regret(cumRegrets, mylabel= "Basic FeedExp3 (anytime)", mycolor = 'blue')
    
    tools.show_plot()

    PricesDist = np.array([0.02, 0.2, 0.01, 0.26, 0.1])
    storage_cost = 2.
    pm_game = games.DynamicPricingPM(PricesDist, storage_cost, feedback_type = 'numeric')
    print "Dynamic Pricing cutomer's prices distribution:", PricesDist
    print
    pm_game.dump()
    tools.show_plot()

    tools.init_plot("FeedExp3 Partial Monitoring Regret on a " + str(len(PricesDist)) + "-levels dynamic pricing problem")
    
    ## Random policy

    print "== Random Baseline =="
    baseline = BasicPolicy(pm_game)
    cumRegrets = tools.eval_policy_parallel(nbCores, nbReps, horizon, pm_game, baseline)
    tools.plot_regret(cumRegrets, mylabel= "Random", mycolor = 'red')

    ## Basic FeedExp3 policy

    print "== BasicFeedExp3 (fixed horizon) =="
    eta, gamma = optimal_Feedexp3_parameters(pm_game, horizon)
    bfx3 = BasicFeedexp3(pm_game, eta , gamma)
    cumRegrets = tools.eval_policy_parallel(nbCores, nbReps, horizon, pm_game, bfx3)
    tools.plot_regret(cumRegrets, mylabel= "Basic FeedExp3 (fixed horizon)", mycolor = 'black')

    ## Basic FeedExp3 policy

    print "== BasicFeedExp3 (anytime) =="
    bfx3 = BasicFeedexp3(pm_game, 0, 0)
    cumRegrets = tools.eval_policy_parallel(nbCores, nbReps, horizon, pm_game, bfx3)
    tools.plot_regret(cumRegrets, mylabel= "Basic FeedExp3 (anytime)", mycolor = 'blue')
    
    
    tools.show_plot()
    


    
