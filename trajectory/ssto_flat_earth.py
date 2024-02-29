#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:42:04 2023

@author: mark
"""

import numpy as np
import openmdao.api as om

from scipy.spatial.transform import Rotation

from trajectory.EOM import QuaternionGroup, FrameRotationsGroup, SixDOFGroup, ForceAndMomentAdderGroup
from propulsion.atmospheric_thrust_comp import AtmThrustComp, ThrottleComp, ThrustVectoringComp
from dymos.models.atmosphere import USatm1976Comp


class SSTOFlatEarthODE(om.Group):
    """
    6 DOF air vehicle over flat earth
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int, desc="Number of nodes to be evaluated in the RHS")

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_subsystem("atmosphere", USatm1976Comp(num_nodes=nn), promotes_inputs=["h"], promotes_outputs=[("pres", "P_amb")])
        self.add_subsystem("throttle", ThrottleComp(num_nodes=nn), promotes_outputs=["*"])
        self.add_subsystem("atmospheric_thrust", AtmThrustComp(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("vector_thrust", ThrustVectoringComp(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])

        self.add_subsystem("quaternion", QuaternionGroup(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])

        to_body = FrameRotationsGroup(num_nodes=nn)
        to_body.add_variable(("g_i", "g_b", "m/s**2"))
        self.add_subsystem(
            "to_body_frame",
            to_body,
            promotes_inputs=[("R_normed", "R_inverse"), "g_i"],
            promotes_outputs=["g_b"],
        )

        summer = ForceAndMomentAdderGroup(num_nodes=nn)
        summer.add_force("r_engine", "F_thrust_b")
        summer.add_force("r_cp", "F_aero_b")

        self.add_subsystem("force_mass_sum", summer, promotes_inputs=["*"], promotes_outputs=["*"])

        self.add_subsystem("EOM", SixDOFGroup(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])

        to_inertial = FrameRotationsGroup(num_nodes=nn)
        to_inertial.add_variable(("w_b", "w_i", "rad/s"))
        to_inertial.add_variable(("V_b", "V_i", "m/s"))
        to_inertial.add_variable(("M", "M_i", "N*m"))
        self.add_subsystem("to_inertial_frame", to_inertial, promotes_inputs=["*"], promotes_outputs=["w_i", "V_i", "M_i"])


if __name__ == "__main__":
    p = om.Problem()
    nn = 5
    p.model.add_subsystem("ode", SSTOFlatEarthODE(num_nodes=nn))

    p.setup(force_alloc_complex=True)

    angles = np.random.uniform(low=-90, high=90, size=[nn, 3])
    rot = Rotation.from_euler("ZYX", angles, degrees=True)
    R = np.roll(rot.inv().as_quat(), 1, axis=1)
    p.set_val("ode.R", val=R)
    p.set_val("ode.V_b", val=np.random.uniform(low=-1000, high=1000, size=[nn, 3]), units="m/s")
    p.set_val("ode.w_b", val=np.random.uniform(low=-10, high=10, size=[nn, 3]), units="rad/s")
    p.set_val("ode.m", val=np.random.uniform(low=100, high=1000, size=nn), units="kg")

    p.set_val("ode.throttle.T_vac", val=10000, units="N")
    p.set_val("ode.throttle.m_dot_no_throttle", val=2.0, units="kg/s")
    p.set_val("ode.A_exit", val=0.1, units="m**2")

    p.run_model()
    p.check_partials(compact_print=True, method="cs", show_only_incorrect=True)
    om.n2(p)
