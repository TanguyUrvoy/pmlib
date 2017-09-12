#!/usr/bin/env python
# coding: utf-8
"""Partial Monitoring Library:
Provides a collection of Finite Partial Monitoring algorithms for experimental studies.
"""
__author__ = "Tanguy Urvoy"
__license__ = "GPL"
__copyright__ = "Orange-labs, France"
__version__ = "1.2"
__email__ = "tanguy.urvoy@orange.com"
__date__ = "2017"
__status__ = "Beta"

import numpy as np
from random import choice

## Basic uniform policy (used as a baseline and example)
## Every new policy should follow this 'startGame/choice/getFeedback' interface

class BasicPolicy:
    def __init__(self, pm):
        self.pm = pm
        self.SelectionProb = np.ones(pm.N, dtype=float)/pm.N
        
    def startGame(self): 
        self.SelectionProb = np.ones(self.pm.N, dtype=float)/self.pm.N

    def choice(self): 
        return np.random.choice(self.pm.N, 1, p = self.SelectionProb)[0]

    def getFeedback(self, action, feedback):
        # do nothing here
        return
