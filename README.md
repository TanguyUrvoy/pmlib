# pmlib
A python library for (finite) Partial Monitoring algorithms

Partial Monitoring(PM) is a general framework for sequential decision making with imperfect feedback.
PM gneralizes a host of problems including for instance: multi-armed bandits, prediction with expert 
advices, dynamic pricing, apple tasting, dark pools, label efficient prediction or dueling bandits.

Each problem is formalized by a couple of NxM matrices L and H.
At each step of the game, the learner chooses an action i and the environment chooses an outome j.
L(i,j) gives the loss of action i for outome j and H(i,j) gives a (symbolic) feedback for this situation.

See N. Cesa-Bianchi, G. Lugosi "Prediction, Learning, and Games" 2006 on chapter 6 for an introduction:
http://homes.dsi.unimi.it/~cesabian/predbook/

This library is based on the the Parma Polyhedra Library for the "Cell decompotion":
http://bugseng.com/products/ppl/
You must intall this library and its python wrapper to use pmlib:
https://pypi.python.org/pypi/pplpy/0.6

We plan to add several generic PM algorithm, but the present version only include FeedExp3 and its variants.
See http://archive.cone.informatik.uni-freiburg.de/pubs/siim-tr-00-18.pdf or 
http://stoltz.perso.math.cnrs.fr/Publications/CBLS-pmonit.pdf

We also provide Rex3, an adhoc algorithm for dueling bandits: 
http://proceedings.mlr.press/v37/gajane15.html

