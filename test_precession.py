#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:54:07 2023

@author: mark
"""

import numpy as np
import openmdao.api as om
import dymos as dm
from trajectory.EOM import SixDOFGroup

# p = om.Problem()
# p.model.add_subsystem("six_dof", SixDOFGroup(num_nodes=10))
# p.setup()
# om.n2(p)

p = om.Problem(model=om.Group())

p.driver = om.pyOptSparseDriver()
p.driver.options["optimizer"] = "IPOPT"
p.driver.declare_coloring()

p.driver.opt_settings["tol"] = 1.0e-5
p.driver.opt_settings["print_level"] = 0
p.driver.opt_settings["mu_strategy"] = "monotone"
p.driver.opt_settings["bound_mult_init_method"] = "mu-based"
p.driver.opt_settings["mu_init"] = 0.01


t = dm.Radau(num_segments=10, order=3)

traj = dm.Trajectory()

phase = dm.Phase(ode_class=SixDOFGroup, transcription=t)
traj.add_phase("phase0", phase)

p.model.add_subsystem("traj", traj)

phase.set_time_options(fix_initial=True, duration_bounds=(1, 10), duration_ref=1)

# velocity in body frame coordinate system
phase.add_state(
    "V_bf",
    fix_initial=True,
    units="m/s",
    rate_source="V_rate",
    # targets= V so we can get X_dot
)

# angular velocity in body frame system
phase.add_state(
    "w_bf",
    fix_initial=True,
    units="rad/s",
    rate_source="w_rate",
    # targets = a quaternion rotatiom thing
)

# body frame to inertial frame rotation quaternion
phase.add_state(
    "R",
    fix_initial=True,
    units="1/s",
    rate_source="R_dot",
    # targets=["R"],
)


# test mixing wildcard ODE variable expansion and unit overrides
# phase.add_timeseries_output(["aero.*", "prop.thrust", "prop.m_dot"], units={"aero.f_lift": "lbf", "prop.thrust": "lbf"})


p.model.linear_solver = om.DirectSolver()

p.setup()

p["traj.phase0.t_initial"] = 0.0
p["traj.phase0.t_duration"] = 350.0

p["traj.phase0.states:r"] = phase.interp("r", [0.0, 111319.54])
p["traj.phase0.states:h"] = phase.interp("h", [100.0, 20000.0])
p["traj.phase0.states:v"] = phase.interp("v", [135.964, 283.159])
p["traj.phase0.states:gam"] = phase.interp("gam", [0.0, 0.0])
p["traj.phase0.states:m"] = phase.interp("m", [19030.468, 16841.431])
p["traj.phase0.controls:alpha"] = phase.interp("alpha", [0.0, 0.0])

dm.run_problem(p, simulate=True)
