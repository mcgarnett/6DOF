#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:39:50 2023

@author: mark
"""
import numpy as np
import openmdao.api as om

from EOM import SixDOFGroup


class Simple6DOF(om.Group):
    """
    just a 6 DOF component, no forces
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int, desc="Number of nodes to be evaluated in the RHS")

    def setup(self):
        nn = self.options["num_nodes"]
