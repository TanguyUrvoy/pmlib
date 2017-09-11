
# coding: utf-8

# In[1]:

import pmlib
import numpy as np


# In[2]:

# uncomment to list all pmlib functionalities
# dir(pmlib)


# In[3]:

# We create a PM instance for a stochatic 3-armed Bernoulli Multi-Armed Bandit (MAB)
# the parameters are the independent expected arm rewards
bandit = pmlib.BernoulliBandit([0.75,0.5,0.25])


# In[4]:

# show the PM game matrices and outcomes distributions
bandit.dump(plot=True, nice=True)

# note that for this example we have 3 actions (one for each arm) 
# and 2^3 outcomes (one for each possible reward vector)


# In[ ]:

# We can also consider bandits with strongly correlated arms
# by specifying a for instance an outcome distribution 
# where only two arms can be winning at the same time
bandit.OutcomeDist = np.array([0,0,0,1/3.,0,1/3.,1/3.,0])
bandit.dump(plot=True, nice=True)


# In[5]:

# Other well known instances of PM are dynamic pricing and Apple tasting

dp = pmlib.DynamicPricingPM([0.1,0.1,0.7,0.1], 2.)
at = pmlib.AppleTasting([0.05,0.95])

at.dump(nice=True)


# In[ ]:

# We can also create a dueling bandit instance where the action is a couple of arms 
# and the feedback is the relative reward.
print "** Dueling bandit problem"    
dueling = pmlib.BinaryUtilityDuelingBanditPM([0.75,0.5,0.5,0.25])
dueling.dump(plot=False,nice=True)


# In[6]:

# We provide a list of benchmark games settings:
# pmlib.benchmark_games


# the pmlib.problemClass() function computes the complexity class of any finite game in the PM hierarchy.
#
# It can be either:
# * trivial     gives \Theta(1) minmax regret
# * easy        gives \Theta(\sqrt(T)) minmax regret
# * hard        gives \Theta(T^{2/3}) minmax regret
# * or intractable with a linear minmax regret
#
# (see Bartok et al. "Partial monitoring – classification, regret bounds, and algorithms" 2013)
# 
# This function and many others are based on the python wrapper to the Parma Polyhedra Library (ppl)
# see http://bugseng.com/products/ppl/
# and https://pypi.python.org/pypi/pplpy/0.6
import ppl

# We can analyze all the games of the benchmark list:

for i in range(len(pmlib.benchmark_games)):
    print
    print
    print "*****", pmlib.benchmark_names[i], "*****"
    game = pmlib.benchmark_games[i]
    game.dump(plot=False, nice = True) # set plot=True to plot the outcome distributions
    # gives game hierarchy
    hierarchy, why = pmlib.ProblemClass(game)
    print
    print
    print "======> This game is", hierarchy.upper() + ", because", why


# In[9]:

# The present version only include 4 variants of FeedExp3 algorithm.
# The BasicFeedexp3 class only works with some forms of numeric feedback matrices where
# there exists an NxN matrix K such that L=KF (See Piccolboni & Schindelhauer 2000).
# The GeneralFeedexp3 uses Cell decomposition to handle more general forms of feedbacks.
# The eta and gamma parameters can be optimized for known horizon.
# When these parameters are set to zero we use an anytime version whith dynamic eta and gamma parameters
# as specified in (Cesa-Bianchi et al. 2006).

from multiprocessing import cpu_count
nbCores = max(1,cpu_count() - 2)
nbReps = nbCores*10
horizon = 10000

pm_game = pmlib.AppleTasting([0.05,0.95])

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [10.,7.]


eta, gamma = pmlib.optimal_Feedexp3_parameters(pm_game, horizon)
gfx3 = pmlib.GeneralFeedexp3(pm_game, eta , gamma)
cumRegrets1 = pmlib.eval_policy_parallel(nbCores, nbReps, horizon, pm_game, gfx3)
pmlib.init_plot("Average regret curve")
pmlib.plot_regret(cumRegrets1, mylabel= "General FeedExp3 (known horizon)", mycolor = 'green')
pmlib.show_plot()
    


# In[10]:

# Here is a generic plot function

def plot_game(pm_game):

    # Feedexp3
    print "** FeedExp3"
    eta, gamma = pmlib.optimal_Feedexp3_parameters(pm_game, horizon)
    gfx3 = pmlib.GeneralFeedexp3(pm_game, eta , gamma)
    cumRegrets1 = pmlib.eval_policy_parallel(nbCores, nbReps, horizon, pm_game, gfx3)

    #Rex3
    if pm_game.game_type=="dueling":
        print "** Rex3"
        rex3 = pmlib.Rex3(pm_game, pmlib.optimal_gamma(pm_game, horizon))
        cumRegrets2 = pmlib.eval_policy_parallel(nbCores, nbReps, horizon, pm_game, rex3)

    # Random
    print "** Random"
    baseline = pmlib.BasicPolicy(pm_game)
    cumRegrets3 = pmlib.eval_policy_parallel(nbCores, nbReps, horizon, pm_game, baseline)

    pmlib.init_plot("Generic PM versus adhoc DB")
    pmlib.plot_regret(cumRegrets1, mylabel= "General FeedExp3 (known horizon)", mycolor = 'black')
    if pm_game.game_type=="dueling":
        # It is an open question whether a general PM algorithm can be as tight as an 
        # adhoc dueling bandits algorithm.
        pmlib.plot_regret(cumRegrets2, mylabel= "Rex3 (known horizon)", mycolor = 'orange')
    pmlib.plot_regret(cumRegrets3, mylabel= "Random", mycolor = 'red', autoscale = False)
    pmlib.show_plot()
    


# In[11]:

# We can also plot all these games (change horizon and nbReps at will)
horizon = 2000
nbReps = nbCores*10

for i in range(len(pmlib.benchmark_games)):
    print
    print
    print "*****", pmlib.benchmark_names[i], "*****"
    game = pmlib.benchmark_games[i]
    print
    plot_game(game)


# In[ ]:



