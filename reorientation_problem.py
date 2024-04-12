#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:04:46 2024

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
from trajectory.reorientation_ode import ODE


p = om.Problem()
p.driver = om.pyOptSparseDriver()
p.driver.options["optimizer"] = "IPOPT"
p.driver.declare_coloring()

# p.driver.opt_settings["tol"] = 1.0e-6
p.driver.opt_settings["max_iter"] = 300
p.driver.opt_settings["print_level"] = 1
p.driver.options["print_results"] = False
# p.driver.opt_settings["mu_strategy"] = "monotone"
# p.driver.opt_settings["bound_mult_init_method"] = "mu-based"
# p.driver.opt_settings["mu_init"] = 1


t = dm.Radau(num_segments=50, order=3)
# t = dm.Birkhoff(grid=dm.BirkhoffGrid(nodes_per_seg=100))

traj = dm.Trajectory()

phase = dm.Phase(ode_class=ODE, transcription=t)
traj.add_phase("phase0", phase)

p.model.add_subsystem("traj", traj)

phase.set_time_options(fix_initial=True, duration_bounds=(1, 60), duration_ref=1)

# mass
# position
# phase.add_state("x", fix_initial=True, units="m", rate_source="V_i")
# velocity in body frame coordinate system
phase.add_state("V_b", fix_initial=True, units="m/s", rate_source="V_b_rate", targets=["V_b"])
# angular velocity in body frame system
phase.add_state("w_b", fix_initial=True, units="rad/s", rate_source="w_b_rate", targets=["w_b"])
# body frame to inertial frame rotation quaternion
phase.add_state("R", fix_initial=True, rate_source="R_rate", targets=["R"])
# gymbal controls
phase.add_control("M_control", opt=True, targets=["M_total_b"], lower=-50, upper=50, units="N*m")


# F-18 mass props
I_mat = np.array(
    [
        [5621, 0, 0],
        [0, 4547, 0],
        [0, 0, 2364],
    ]
)

# SSTO moment of inertia
phase.add_parameter("I", val=I_mat, units="kg*m**2", targets=["I"])

# test mixing wildcard ODE variable expansion and unit overrides
# phase.timeseries_options["include_parameters"] = False
phase.add_timeseries_output(["w_i", "w_b", "M_i", "X_i", "Y_i", "Z_i", "M_control"])
# phase.add_timeseries_output("*")
phase.add_objective("time", loc="final", scaler=1)

phase.add_boundary_constraint("R_normed", loc="final", equals=np.array([1, 0, 0, 0]))

# p.model.linear_solver = om.DirectSolver()
p.driver.declare_coloring()
p.setup()

p["traj.phase0.t_initial"] = 0.0
p["traj.phase0.t_duration"] = 40.0
# p["traj.phase0.states:m"] = phase.interp("m", [m_init, m_fin])
# p["traj.phase0.states:x"] = phase.interp("x", [0, 1e6])
p["traj.phase0.states:w_b"] = phase.interp("w_b", [[0, 0, 0], [0, 0.0, 0.0]])
p["traj.phase0.states:V_b"] = phase.interp("V_b", [[0, 0, 0], [0, 0, 0]])

# initial inertial to body frame rotation. Z yaw, Y pitch, X roll
# rot = Rotation.from_euler("ZYX", [0, 90, 0], degrees=True)
rot = Rotation.from_euler("ZYX", [0, 45, 45], degrees=True)
# rot = Rotation.from_euler("ZYX", [23, 45, -45], degrees=True)
# invert it for body to inertial, turn into quat intp format with w first instead of last
p["traj.phase0.states:R"] = np.roll(rot.inv().as_quat(), 1)

# dm.run_problem(p, run_driver=False, simulate=False, make_plots=False)
dm.run_problem(p, run_driver=True, simulate=False, make_plots=True)
# sim = p.model.traj.simulate(record_file="precession_sim.db")
om.n2(p)


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[20, 20])
ax1, ax2, ax3, ax4 = axs.flatten()

t = p.get_val("traj.phase0.timeseries.time")
w_i = p.get_val("traj.phase0.timeseries.w_i")
w_b = p.get_val("traj.phase0.timeseries.w_b")
M_b = p.get_val("traj.phase0.timeseries.M_control")
M_i = p.get_val("traj.phase0.timeseries.M_i") /50
X_i = p.get_val("traj.phase0.timeseries.X_i")
Y_i = p.get_val("traj.phase0.timeseries.Y_i")
Z_i = p.get_val("traj.phase0.timeseries.Z_i")
w_i_normed = w_i / np.linalg.norm(w_i, axis=1)[:, np.newaxis]

ax1.plot(t, w_i, label="w inertial")
ax3.plot(t, X_i, label="body X axis")
ax2.plot(t, w_b, label="w body")
ax4.plot(t, M_b, label="M control")
ax1.legend()
ax2.legend()
ax3.legend()

# ax3 = plt.figure(figsize=[10, 10]).add_subplot(projection="3d")
# ax3.quiver(0, 0, 0, w_i[:, 0], w_i[:, 1], w_i[:, 2], color="b")
# # ax3.quiver(0, 0, 0, w_b[:, 0], w_b[:, 1], w_b[:, 2],color="r")
# # ax3.quiver(0,0,0,1,1,1)

# ax3.set_xlim(-1, 1)
# ax3.set_ylim(-1, 1)
# ax3.set_zlim(-1, 1)

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=[15, 15])

Q4 = ax.quiver(0, 0, 0, M_i[0, 0], M_i[0, 1], M_i[0, 2], color="m")
Q1 = ax.quiver(0, 0, 0, X_i[0, 0], X_i[0, 1], X_i[0, 2], color="r")
Q2 = ax.quiver(0, 0, 0, Y_i[0, 0], Y_i[0, 1], Y_i[0, 2], color="g")
Q3 = ax.quiver(0, 0, 0, Z_i[0, 0], Z_i[0, 1], Z_i[0, 2], color="b")
line = ax.plot([], [], [], lw=2, color="m")[0]
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

def update_quiver(frame):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    global Q1
    global Q2
    global Q3
    global Q4
    global line
    # Q.remove()
    Q1.remove()
    Q2.remove()
    Q3.remove()
    Q4.remove()
    Q4 = ax.quiver(0, 0, 0, M_i[frame, 0], M_i[frame, 1], M_i[frame, 2], color="m")
    Q1 = ax.quiver(0, 0, 0, X_i[frame, 0], X_i[frame, 1], X_i[frame, 2], color="r")
    Q2 = ax.quiver(0, 0, 0, Y_i[frame, 0], Y_i[frame, 1], Y_i[frame, 2], color="g")
    Q3 = ax.quiver(0, 0, 0, Z_i[frame, 0], Z_i[frame, 1], Z_i[frame, 2], color="b")
    line.set_data(w_i_normed[0:frame, :2].T)
    line.set_3d_properties(w_i_normed[:frame, 2])


# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, frames=w_i.shape[0], interval=50, blit=False)
plt.show()
