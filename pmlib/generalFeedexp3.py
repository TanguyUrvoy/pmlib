#!/usr/bin/env python
# coding: utf-8
"""Partial Monitoring Library:
Provides a collection of Finite Partial Monitoring algorithms for experimental studies.
"""


__author__ = "Tanguy Urvoy, Pratik Gajane"
__copyright__ = "Orange-labs, France"
__license__ = "GPL"
__version__ = "1.2"
__email__ = "tanguy.urvoy@orange.com, pratik.gajane@gmail.com"
__date__ = "2017"
__status__ = "Beta"

import numpy as np
import games
import pmcells
import random
import copy

#
# Transform the matrices in order to apply the basic FeedExp algorithm (General FeedExp3 helper)
# 
def transform_matrices(FeedbackMatrix, LossMatrix):
    
    assert FeedbackMatrix.shape == LossMatrix.shape
    N, M = FeedbackMatrix.shape

    #
    # Fist step: build the F' and L' matrices from F and L
    #

    Fdash_list = []
    Ldash_list = []
    # We use the lists sizes for z in paper
    h = {} # pseudo-action to action map
    s = {} # pseudo-action to symbol map

    # for each action i
    for i in range(N):

        # a=0 in the paper page 21
        bool_fiv_added = False
    
        # for each value in the feedback alphabet related to that action i
        for v in set(FeedbackMatrix[i,...]):
        
            # build the signal vector Fiv  i.e. [F_{i,j}=v]_{j=1,...,M} in the the paper
            Fiv = pmcells.signal_vec(i,v,FeedbackMatrix)

            # Constructing Fdash. i.e. F' in the paper
            # if Fdash_list is empty or if it's not redundant with previously added pseudo-actions then
            # we add the signal vector
            if not Fdash_list or not pmcells.is_linear_comb(Fiv, Fdash_list):
                h[len(Fdash_list)] = i # h(z)=i in FeedExp3 paper
                s[len(Fdash_list)] = v # not in FeedExp3 paper ??
                Fdash_list.append(Fiv)
                Ldash_list.append(LossMatrix[i,...])
                bool_fiv_added = True

        # if action i did not lead to a new feedback we add an "empty feedback" i.e. zero line in the matrix
        if not bool_fiv_added:
            h[len(Fdash_list)] = i # h(z)=j in the paper
            s[len(Fdash_list)] = v # not in FeedExp3 paper ??
            Fdash_list.append(np.zeros(M))
            Ldash_list.append(LossMatrix[i,...])
        
    # Build F' and H' matrices
    FdashMatrix = np.vstack(Fdash_list)
    LdashMatrix = np.vstack(Ldash_list)

    Ndash, Mdash = FdashMatrix.shape
    assert FdashMatrix.shape == LdashMatrix.shape # just in case
 
    #
    # Second step:
    #
        
    # Search for strictly-dominating pseudo-actions 
    NonEmptyCells = []
    EmptyCells = []
    for iv in range(Ndash):
        if pmcells.isStrictlyNonDominated(iv, LdashMatrix):
            NonEmptyCells.append(iv)
        else:
            EmptyCells.append(iv)
            
    if len(NonEmptyCells)>0: # An empty nonEmptyCells is a problem!
        # Pick one non-dominated action
        b = random.choice(NonEmptyCells)      # Choose any action from the set of actions with nonempty cells
    else:
        print "WARNING: no strictly dominant cell found"
        #b = random.choice(range(Ndash))  # Choose any action
        b = 0

    # Translate the loss relatively to pseudo-action b. Recall that loss transposition does not impact the policy regret.
    LdashMatrix = LdashMatrix - LdashMatrix[b,...]
    
    # Makes the dominated actions as bad as possible i.e. with worst possible loss
    for iv in EmptyCells:
        if not pmcells.isNonDominated(iv, LdashMatrix):
            LdashMatrix[iv,...] =  max(LdashMatrix[iv,...])
            

    return FdashMatrix,LdashMatrix,h,s



## this debug function checks the general Feedexp3 matrices transformation on a given pm game

def check_matrices_transformation(pm):
    pm.dump(plot=True)
    FeedbackMatrix = pm.FeedbackMatrix
    LossMatrix = pm.LossMatrix
    N, M = pm.N, pm.M
    OutcomeDist = pm.OutcomeDist

    print "Before: outcomes M=" + str(M) + " actions N=" + str(N)
    print "Feedback matrix:"
    print (FeedbackMatrix)
    print "Loss matrix:"
    print (LossMatrix)
    print

    # Prepare matrices for basic FeedExp3
    FdashMatrix,LdashMatrix,h,s = pmlib.transform_matrices(FeedbackMatrix, LossMatrix)

    print "After: outcomes M=" + str(M) + " pseudo-actions N'=" + str(len(h))
    print "Feedback/signal matrix:"
    print (FdashMatrix)
    print "Loss matrix:"
    print (LdashMatrix)

    print
    print "Pseudo-action to action,symbol translation:"
    for iv in range(len(h)):
        print "speudo-action:", iv, "action:", h[iv], "symbol:", s[iv]




from basicFeedexp3 import BasicFeedexp3


# General FeedExp3 algorithm
# As described in [Piccolboni Schindelhauer "Discrete Prediction Games with Arbitrary Feedback and Loss" 2000] (but with NxM matrices)
# and [Bianchi et al. 2006 "Regret minimization under partial monitoring"]
# First transform the PM problem and wrap basicFeedexp3 alorithm

class GeneralFeedexp3(BasicFeedexp3):
    def __init__(self, pm, eta = 0, gamma = 0):

        # print "NxM={0}x{1}".format(pm.N, pm.M)
        # print "general-LossMatrix="
        # print pm.LossMatrix
        # print "general-FeedbackMatrix="
        # print pm.FeedbackMatrix
        # print

        # Prepare matrices and game for basic FeedExp3
        Fdash, Ldash, self.real_action, self.real_symbol = transform_matrices(pm.FeedbackMatrix, pm.LossMatrix)
        pm2 = copy.deepcopy(pm)
        pm2.N, pm2.M = Ldash.shape
        pm2.FeedbackMatrix, pm2.LossMatrix = Fdash, Ldash
        
        pm2.Actions_dict = { i : (self.real_action[i], self.real_symbol[i]) for i in range(pm2.N)}

        # initialize basic FeedExp3 for this new 'internal' game
        BasicFeedexp3.__init__(self, pm2, eta, gamma)
        

    # Feedexp3 choice wrapper
    def choice(self):
        self.previous_iv = BasicFeedexp3.choice(self)
        return self.real_action[self.previous_iv]

    # Feedexp3 feedback wrapper
    def getFeedback(self, action, symbolic_feedback):
        assert action<self.N

        assert self.real_action[self.previous_iv] == action # sanity check

        # one possible approch is to compare expected symbol:
        expected_symbol = self.real_symbol[self.previous_iv]

        if expected_symbol == symbolic_feedback:
            BasicFeedexp3.getFeedback(self, self.previous_iv, 1.)
        else:
            BasicFeedexp3.getFeedback(self, self.previous_iv, 0.)

        
