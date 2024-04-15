#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:42:04 2023

@author: mark
"""

import numpy as np
import openmdao.api as om

from trajectory.EOM import QuaternionGroup, FrameRotationsGroup, SixDOFGroup, ForceAndMomentAdderGroup


class ODE(om.Group):
    """
    just a 6 DOF component, no forces
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int, desc="Number of nodes to be evaluated in the RHS")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_subsystem("quaternion", QuaternionGroup(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])

        to_body = FrameRotationsGroup(num_nodes=nn)
        to_body.add_variable(("g_i", "g_b", "m/s**2"))
        self.add_subsystem(
            "to_body_frame",
            to_body,
            promotes_inputs=[("R_normed", "R_inverse"), "g_i"],
            promotes_outputs=["g_b"],
        )

        self.add_subsystem("EOM", SixDOFGroup(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])

        to_inertial = FrameRotationsGroup(num_nodes=nn)
        to_inertial.add_variable(("w_b", "w_i", "rad/s"))
        to_inertial.add_variable(("V_b", "V_i", "m/s"))
        to_inertial.add_variable(("M_total_b", "M_i", "N*m"))
        to_inertial.add_variable(("X_b", "X_i", "m"))
        to_inertial.add_variable(("Y_b", "Y_i", "m"))
        to_inertial.add_variable(("Z_b", "Z_i", "m"))
        to_inertial.add_variable(("g_b", "g_i_2", "m/s**2"))
        self.add_subsystem("to_inertial_frame", to_inertial, promotes_inputs=["*"], promotes_outputs=["*"])

        self.set_input_defaults("X_b", val=np.tile(np.array([1, 0, 0]), (nn, 1)))
        self.set_input_defaults("Y_b", val=np.tile(np.array([0, 1, 0]), (nn, 1)))
        self.set_input_defaults("Z_b", val=np.tile(np.array([0, 0, 1]), (nn, 1)))

        self.set_input_defaults("g_i", val=np.tile(np.array([0, 0, -1]), (nn, 1)))


if __name__ == "__main__":
    p = om.Problem()

    p.model.add_subsystem("ode", ODE(num_nodes=20))

    p.setup()
    om.n2(p)
