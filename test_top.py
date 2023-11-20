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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.quiver import Quiver

# from dm.visualization.timeseries_plots import timeseries_plots
from trajectory.spinning_top import SpinningTopODE


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


t = dm.Radau(num_segments=30, order=3)
# t = dm.Birkhoff(grid=dm.BirkhoffGrid(num_segments=1, nodes_per_seg=50, grid_type="lgl"))
traj = dm.Trajectory()

phase = dm.Phase(ode_class=SpinningTopODE, transcription=t)
traj.add_phase("phase0", phase)

p.model.add_subsystem("traj", traj)

phase.set_time_options(fix_initial=True, duration_bounds=(1, 20), duration_ref=1)

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
I_mat = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 10]])
phase.add_parameter("F_by_m", units="m/s**2", targets=["F_by_m"])

phase.add_parameter("F_grav_i", units="N", val=np.array([0, 0, -10]), targets=["to_body_frame.F_grav_i"])

phase.add_parameter("I", val=I_mat, units="kg*m**2", targets=["I"])

# test mixing wildcard ODE variable expansion and unit overrides
phase.add_timeseries_output(["w_i", "w_b", "M_i"])

phase.add_objective("time", loc="final", scaler=-1)
# traj.add_parameter("F_by_m", val=np.zeros(3), units="m/s**2", opt=False)
# traj.add_parameter("M", val=np.zeros(3), units="N*m", opt=False)

# p.model.linear_solver = om.DirectSolver()
p.driver.declare_coloring()
p.setup()
p.set_val("traj.phase0.parameters:F_by_m", val=np.array([0, 0, 0]))
p["traj.phase0.t_initial"] = 0.0
p["traj.phase0.t_duration"] = 10.0


p["traj.phase0.states:w_b"] = phase.interp("w_b", [[0, 0, 2], [0, 0.0, 0.0]])
p["traj.phase0.states:V_b"] = phase.interp("V_b", [[0, 0, 0], [0, 0, 0]])
# initial inertial to body frame rotation. Z yaw, Y pitch, X roll
rot = Rotation.from_euler("ZYX", [0, 10, 0], degrees=True)
# invert it for body to inertial, turn into quat intp format with w first instead of last
p["traj.phase0.states:R"] = np.roll(rot.inv().as_quat(), 1)

# p.run_model()
dm.run_problem(p, simulate=False, make_plots=False)
# dm.visualization.timeseries_plots.timeseries_plots(simulation_record_file="precession_sim.db")
# sim = p.model.traj.simulate(record_file="precession_sim.db")
om.n2(p)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[20, 10])
ax1, ax2 = axs

t = p.get_val("traj.phase0.timeseries.time")
w_i = p.get_val("traj.phase0.timeseries.w_i")
w_b = p.get_val("traj.phase0.timeseries.w_b")
M_i = p.get_val("traj.phase0.timeseries.M_i")
w_i_normed = w_i / np.linalg.norm(w_i, axis=1)[:, np.newaxis]

ax1.plot(t, w_i, label="w inertial")
ax2.plot(t, w_b, label="w body")
ax1.legend()
ax2.legend()


# ax3 = plt.figure(figsize=[10, 10]).add_subplot(projection="3d")
# ax3.quiver(0, 0, 0, w_i[:, 0], w_i[:, 1], w_i[:, 2], color="b")
# # ax3.quiver(0, 0, 0, w_b[:, 0], w_b[:, 1], w_b[:, 2],color="r")
# # ax3.quiver(0,0,0,1,1,1)

# ax3.set_xlim(-1, 1)
# ax3.set_ylim(-1, 1)
# ax3.set_zlim(-1, 1)

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=[15, 15])

Q1 = ax.quiver(0, 0, 0, w_i[0, 0], w_i[0, 1], w_i[0, 2], color="b")
Q2 = ax.quiver(0, 0, 0, M_i[0, 0], M_i[0, 1], M_i[0, 2], color="r")
line = ax.plot([], [], [], lw=2, color="m")[0]
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)


def update_quiver(frame):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    global Q1
    global Q2
    global line
    # Q.remove()
    Q1.remove()
    Q2.remove()
    Q1 = ax.quiver(0, 0, 0, w_i[frame, 0], w_i[frame, 1], w_i[frame, 2], color="b")
    Q2 = ax.quiver(0, 0, 0, M_i[frame, 0], M_i[frame, 1], M_i[frame, 2], color="r")
    line.set_data(w_i_normed[0:frame, :2].T)
    line.set_3d_properties(w_i_normed[:frame, 2])


# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, frames=w_i.shape[0], interval=50, blit=False)
plt.show()
