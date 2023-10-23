#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:54:07 2023

@author: mark
"""

import numpy as np
import openmdao.api as om
import dymos as dm
from scipy.spatial.transform import Rotation

# from dm.visualization.timeseries_plots import timeseries_plots
from trajectory.EOM import SixDOFGroup

# p = om.Problem()
# p.model.add_subsystem("six_dof", SixDOFGroup(num_nodes=10))
# p.setup()
# om.n2(p)

# p = om.Problem(openmdao_reports_dir="reports/precession/")
p = om.Problem()
p.driver = om.pyOptSparseDriver()
p.driver.options["optimizer"] = "IPOPT"
p.driver.declare_coloring()

# p.driver.opt_settings["tol"] = 1.0e-6
p.driver.opt_settings["print_level"] = 1
p.driver.opt_settings["mu_strategy"] = "monotone"
# p.driver.opt_settings["bound_mult_init_method"] = "mu-based"
# p.driver.opt_settings["mu_init"] = 1


t = dm.Radau(num_segments=10, order=3)

traj = dm.Trajectory()

phase = dm.Phase(ode_class=SixDOFGroup, transcription=t)
traj.add_phase("phase0", phase)

p.model.add_subsystem("traj", traj)

phase.set_time_options(fix_initial=True, duration_bounds=(1, 10), duration_ref=1)

# velocity in body frame coordinate system
phase.add_state(
    "V_b",
    # shape=(3,),
    fix_initial=True,
    units="m/s",
    rate_source="V_b_rate",
    targets=["V_b"],
)

# angular velocity in body frame system
phase.add_state(
    "w_b",
    # shape=(3,),
    fix_initial=True,
    units="rad/s",
    rate_source="w_b_rate",
    targets=["w_b"],
)


# body frame to inertial frame rotation quaternion
phase.add_state(
    "R",
    fix_initial=True,
    rate_source="R_rate",
    targets=["R"],
)
I_mat = np.array([10, 0, 0], [0, 1, 0], [0, 0, 1])
phase.add_parameter("F_by_m", units="m/s**2", targets=["F_by_m"])
phase.add_parameter("M", units="N*m", targets=["M"])
phase.add_parameter("I", val=I_mat, units="kg*m**2", targets=["I"])

# test mixing wildcard ODE variable expansion and unit overrides
# phase.add_timeseries_output("*")

phase.add_objective("time", loc="final", scaler=-1)
# traj.add_parameter("F_by_m", val=np.zeros(3), units="m/s**2", opt=False)
# traj.add_parameter("M", val=np.zeros(3), units="N*m", opt=False)

# p.model.linear_solver = om.DirectSolver()
p.driver.declare_coloring()
p.setup()
p.set_val("traj.phase0.parameters:F_by_m", val=np.array([0, 0, -0.1]))
p.set_val("traj.phase0.parameters:M", val=np.array([-0.1, 0, 0]))
p["traj.phase0.t_initial"] = 0.0
p["traj.phase0.t_duration"] = 10.0


p["traj.phase0.states:w_b"] = phase.interp("w_b", [[0.5, 0.5, 0], [0.5, 0.5, 0]])
p["traj.phase0.states:V_b"] = phase.interp("V_b", [[0, 0, 1], [0, 0, 1]])
rot = Rotation.from_euler("XYZ", [90, 0, 0], degrees=True)
p["traj.phase0.states:R"] = np.roll(rot.as_quat(), 1)
# p["traj.phase0.states:m"] = phase.interp("m", [19030.468, 16841.431])
# p["traj.phase0.controls:alpha"] = phase.interp("alpha", [0.0, 0.0])

# p.run_model()
dm.run_problem(p, simulate=False, make_plots=False)
# dm.visualization.timeseries_plots.timeseries_plots(simulation_record_file="precession_sim.db")
# sim = p.model.traj.simulate(record_file="precession_sim.db")
om.n2(p)
