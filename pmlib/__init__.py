#!/usr/bin/env python
# coding: utf-8
"""Partial Monitoring Library:
Provides a collection of (Finite) Partial Monitoring(PM) algorithms for experimental studies.
"""
__author__ = "Tanguy Urvoy"
__copyright__ = "Orange-labs, France"
__license__ = "GPL"
__version__ = "1.2"
__email__ = "tanguy.urvoy@orange.com"
__date__ = "2017"
__status__ = "Beta"


# games include a collection of PM games definitions 
from games import *

# policies include a baseline random policy and other PM algorithms
from policies import BasicPolicy
from pmcells import *
from basicFeedexp3 import BasicFeedexp3, optimal_Feedexp3_parameters
from generalFeedexp3 import transform_matrices, GeneralFeedexp3, check_matrices_transformation

# adhoc policies
from rex3 import *


# tools include parallelized routines for stochastic environement simulation and
# regret curves plotting
from tools import *
