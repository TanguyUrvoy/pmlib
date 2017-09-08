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
import games

## Convex polyhedron manipulation library
import ppl



### General FeedExp3 helpers

# build the signal vector i.e. [F_{i,j}=v]_{j=1,...,M}
# (with NxM format matrices)

def signal_vec(i,v,FeedbackMatrix):
    return (FeedbackMatrix[i,...] == v).astype(np.float)

# build the signal matrix  i.e. [F_{i,j}=v]_{j=1,...,M}

def signal_mat(i,FeedbackMatrix):
    return np.stack([signal_vec(i,v,FeedbackMatrix) for v in set(FeedbackMatrix[i,...])])


if __name__ == "__main__":
    pm = games.BinaryUtilityDuelingBanditPM([0.9,0.5,0.5,0.25], feedback_type = 'string')      
    Fmat = pm.FeedbackMatrix
    Lmat = pm.LossMatrix

    print Fmat[1,:]
    print signal_mat(1,Fmat)

    
# Fjv: the row binary signal matrix for F_{i,j}=v (pseudo-action i/v)
# Fdash_list : a list of row feedback vectors
# check if a signal matrix is in the linear combination of the previous feedback vectors
# i.e. that staking this new vector does not increase matrix rank

def is_linear_comb(Fiv, Fdash_list):
    if len(Fdash_list) == 0:
        return True
    initial_rank = np.linalg.matrix_rank(np.vstack(Fdash_list))
    new_rank = np.linalg.matrix_rank(np.vstack((Fdash_list,Fiv)))
    return new_rank == initial_rank

if __name__ == "__main__":
    # simple test
    test_list = [np.array([1,1,0]),np.array([0,1,0])]
    assert is_linear_comb(np.array([1,0,0]), test_list)
    assert not is_linear_comb(np.array([0,0,1]), test_list)


## Domination Cells decomposition of a Partial Monitoring Game


# Domination matrix is the upper bound constraint for the i-th action's PM Cell
# (LossMatrix[i,...] - LossMatrix).dot(p) < 0 means that 
# i is the best action for outcome distribution p
def domination_matrix(i,LossMatrix):
    return LossMatrix[i,...] - LossMatrix

# transform a floating point Domination matrix into an equivalent integer matrix
def scale_to_integers(Dom):
    where = np.modf(Dom)[0] != 0
    if where.any():
        #print "WARNING: ppl works only with integers and silently removes frational part of floats"
        #print "Rescaling Domination Matrix!"
        m = np.min(Dom[where])
        return Dom/np.abs(m)
    return Dom

        


# return domination Cell polytope for action i
def DominationPolytope(i,LossMatrix):
    N, M = LossMatrix.shape

    # declare M ppl Variables
    p = [ppl.Variable(j) for j in range(M)]
    
    # declare polytope constraints
    cs = ppl.Constraint_System()
    
    # probabilies constraints on p
    cs.insert( sum( p[j] for j in range(M)) == 1 )
    for j in range(M):
        cs.insert(p[j] >= 0)
        
    # strict Loss domination constraints
    Dom = scale_to_integers(domination_matrix(i,LossMatrix))
    
    for a in range(N):
        if a != i:
            # p is such that for any action a Loss[i,...]*p <= Loss[a,...]*p
            #print "Domination line:", Dom[a,...], "inequality:", sum( (Dom[a,j]*p[j] for j in range(M)) ) <= 0
            cs.insert( sum( (Dom[a,j]*p[j] for j in range(M)) ) <= 0 )
            
    return ppl.C_Polyhedron(cs)


# return domination Cell polytope interior for action i
def StrictDominationPolytope(i,LossMatrix):
    N, M = LossMatrix.shape

    # declare M ppl Variables
    p = [ppl.Variable(j) for j in range(M)]
    
    # declare polytope constraints
    cs = ppl.Constraint_System()
    
    # probabilies constraints on p
    cs.insert( sum( p[j] for j in range(M)) == 1 )
    for j in range(M):
        cs.insert(p[j] >= 0)
        
    # strict Loss domination constraints
    Dom = scale_to_integers(domination_matrix(i,LossMatrix))    

    for a in range(N):
        if (Dom[a,...] != 0).any():
            # p is such that for any action a Loss[i,...]*p <= Loss[a,...]*p
            #print "Strict Domination line:", Dom[a,...], "inequality:", sum( (Dom[a,j]*p[j] for j in range(M)) ) < 0
            cs.insert( sum( (Dom[a,j]*p[j] for j in range(M)) ) < 0 )
            
    return ppl.NNC_Polyhedron(cs)

# Check that an action is dominant
# Check that an action is strictly dominant 
# i.e. there exists some outcome distributions where i is one of the best actions
def isNonDominated(i, LossMatrix):
    return not (DominationPolytope(i,LossMatrix).is_empty())

# Check that an action is strictly dominant 
# i.e. there exists some outcome distributions where i is strictly the best action
def isStrictlyNonDominated(i, LossMatrix):
    return not (StrictDominationPolytope(i,LossMatrix).is_empty())

# Check if an action is degenerated
# i.e. if there exists another action Cell containing strictly its cell.
def isDegenerated(i,LossMatrix):
    N,M = LossMatrix.shape
    polytope_i = DominationPolytope(i, LossMatrix)
    if polytope_i.is_empty():
        return False
    isDegen = False
    j=0
    while(not isDegen and j<N):
        if j!=i:
            # strict inclusion test
            if polytope_i < DominationPolytope(j, LossMatrix):
                #print "Cell(",i,") is strictly inside Cell(", j, ")"
                isDegen = True
        j += 1
    return isDegen       

# Check if an action is pareto optimal
def isParetoOptimal(i, LossMatrix):
    return isNonDominated(i, LossMatrix) and not isDegenerated(i,LossMatrix)


# Return the polytope where both a and b are winning actions
def interFacePolytope(a, b, LossMatrix):
    N, M = LossMatrix.shape

    # declare M ppl Variables
    p = [ppl.Variable(j) for j in range(M)]
    
    # declare polytope constraints
    cs = ppl.Constraint_System()
    
    # probabilies constraints on p
    cs.insert( sum( p[j] for j in range(M)) == 1 )
    for j in range(M):
        cs.insert(p[j] >= 0)
        
    # strict Loss domination constraints for both a and b
    Doma = scale_to_integers(domination_matrix(a,LossMatrix))
    Domb = scale_to_integers(domination_matrix(b,LossMatrix))        
    for i in range(N):
        if i!=a:
            # p is such that for any action i Loss[a,...]*p <= Loss[a,...]*p
            cs.insert( sum( (Doma[i,j]*p[j] for j in range(M)) ) <= 0 )
        if i!=b:
            # p is such that for any action i Loss[b,...]*p <= Loss[a,...]*p
            cs.insert( sum( (Domb[i,j]*p[j] for j in range(M)) ) <= 0 )
            
    return ppl.C_Polyhedron(cs)

# Check if two actions are neighbours
def areNeighbours(a, b, LossMatrix):
    return interFacePolytope(a, b, LossMatrix).affine_dimension() >= M - 2


# Returns the neigbourhood of a pair of actions
def Neighbourhood(a, b, LossMatrix):
    N, M = LossMatrix.shape
    
    frontier = interFacePolytope(a, b, LossMatrix)
    
    Nb = []
    for k in range(N):
        if k==a or k==b or frontier <= DominationPolytope(k,LossMatrix):
            Nb.append(k)
    
    return Nb


if __name__ == "__main__":
    # test domination test
    Lmat = np.array([[2, 0, 0, 2],
                     [2, 0, 0, 0],
                     [2, 2, 2, 2]])

    print "Loss matrix:"
    print Lmat
    for i in range(Lmat.shape[0]):
        print 
        print "Domination matrix for action", i, ":"
        print domination_matrix(i,Lmat)
        print "Domination polytope:", DominationPolytope(i,Lmat).minimized_generators()
        print "dominating:", isNonDominated(i,Lmat)
        print "Strict Domination polytope:", StrictDominationPolytope(i,Lmat).minimized_generators()
        print "strictly dominating:", isStrictlyNonDominated(i,Lmat)
        print "degenerated:", isDegenerated(i,Lmat)
        print "Pareto Optimal:", isParetoOptimal(i, Lmat)
    


        pmg = games.BernoulliBandit([0.75,0.5,0.25], feedback_type = 'string')
        Lmat = pmg.LossMatrix
                                   
        print "Loss matrix:"
        print Lmat

        N, M = Lmat.shape

        print "global dim:", M

        for a in range(N):
            print a, " dim=", DominationPolytope(a,Lmat).affine_dimension()
            for b in range(a+1,N):
                pol = interFacePolytope(a, b, Lmat)
                dim = pol.affine_dimension()
                print pmg.Actions_dict[a],pmg.Actions_dict[b], dim, pol.is_empty()

                




# the end
