#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:42:04 2023

@author: mark
"""

import numpy as np
import openmdao.api as om

from trajectory.EOM import QuaternionGroup, FrameRotationsGroup, SixDOFGroup


class SpinningTopODE(om.Group):
    """
    just a 6 DOF component, no forces
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int, desc="Number of nodes to be evaluated in the RHS")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_subsystem("quaternion", QuaternionGroup(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])

        to_body = FrameRotationsGroup(num_nodes=nn)
        to_body.add_variable(("F_grav_i", "F_grav_b", None))
        self.add_subsystem(
            "to_body_frame",
            to_body,
            promotes_inputs=[("R_normed", "R_inverse")],
            promotes_outputs=["F_grav_b"],
        )

        # calculate torque

        cross_prod_1 = om.CrossProductComp(
            c_name="M",
            a_name="F_grav_b",
            b_name="x_b_axis",
            c_units="N*m",
            a_units="N",
            b_units="m",
            vec_size=nn,
        )

        self.add_subsystem("grav_torque", cross_prod_1, promotes_inputs=["*"], promotes_outputs=["M"])

        self.add_subsystem("EOM", SixDOFGroup(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])

        to_inertial = FrameRotationsGroup(num_nodes=nn)
        to_inertial.add_variable(("w_b", "w_i", "rad/s"))
        to_inertial.add_variable(("V_b", "V_i", "m/s"))
        to_inertial.add_variable(("M", "M_i", "N*m"))
        self.add_subsystem(
            "to_inertial_frame", to_inertial, promotes_inputs=["*"], promotes_outputs=["w_i", "V_i", "M_i"]
        )

        x_b_axis = np.tile(np.array([0, 0, 1]), [nn, 1])
        self.set_input_defaults("x_b_axis", val=x_b_axis)


if __name__ == "__main__":
    p = om.Problem()

    p.model.add_subsystem("ode", SpinningTopODE(num_nodes=20))

    p.setup()
    om.n2(p)
