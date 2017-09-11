# pmlib
A python library for (finite) Partial Monitoring algorithms

## Partial monitoring

Partial Monitoring(PM) is a general framework for sequential decision making with imperfect feedback.
PM generalizes a host of problems including for instance multi-armed bandits, prediction with expert 
advices, dynamic pricing, apple tasting, dark pools, label efficient prediction and dueling bandits.

Each problem is formalized by a couple of NxM matrices L and F.
At each step of the game, the learner chooses an action i and the environment chooses an outome j.
L(i,j) gives the loss of action i for outome j and F(i,j) gives a (symbolic or numeric) feedback for this situation.

See N. Cesa-Bianchi, G. Lugosi "Prediction, Learning, and Games" 2006 on chapter 6 for an introduction:
http://homes.dsi.unimi.it/~cesabian/predbook/

## The library

We plan to add several generic PM algorithm, but the present version only includes FeedExp3 and its variants.
See http://archive.cone.informatik.uni-freiburg.de/pubs/siim-tr-00-18.pdf or 
http://stoltz.perso.math.cnrs.fr/Publications/CBLS-pmonit.pdf

We also provide Rex3, an adhoc algorithm for dueling bandits: 
http://proceedings.mlr.press/v37/gajane15.html


The Function `pmlib.problemClass(game)` can analyze any game and provide its position in the PM complexity hierarchy as defined in
(Bartok et al. "Partial monitoring – classification, regret bounds, and algorithms" 2013).
This can be either:
* **TRIVIAL** if a single action domines all others. It gives a constant regret.
* **EASY** if it is hard but all neighbouring actions are locally observable. It gives a regret in \sqrt(T).
* **HARD** if all pairs of actions are globally observable. It gives a regret in T^{2/3}.
* **INTRACTABLE** if some pairs of actions are non globally observable.

## install guide
This library is based on the the Parma Polyhedra Library for the "Cell decomposition":
* http://bugseng.com/products/ppl/

You must intall this library and its python wrapper to use pmlib:
* https://pypi.python.org/pypi/pplpy/0.6

We also use numpy, scipy and pandas.
* http://www.numpy.org/
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.logsumexp.html
* http://pandas.pydata.org/

