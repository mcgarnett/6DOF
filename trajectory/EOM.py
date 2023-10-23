#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 08:34:12 2023

@author: mark
"""

import openmdao.api as om
import numpy as np


class SixDOFGroup(om.Group):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

        # self.options.declare("r0", types=float, default=6.3781e6, desc="Earth radius m")

        # self.options.declare("g0", types=float, default=9.80665, desc="acceleration at earth surface")

    def setup(self):
        nn = self.options["num_nodes"]

        mat_vec_prod = om.MatrixVectorProductComp(
            A_name="I",
            A_units="kg*m**2",
            x_name="w_b",
            x_units="rad/s",
            b_name="I_times_w_b",
            b_units="kg*m**2/s",
            vec_size=nn,
            A_shape=(3, 3),
        )

        cross_prod_1 = om.CrossProductComp(
            c_name="w_b_cross_V_b",
            a_name="w_b",
            b_name="V_b",
            c_units="m/s**2",
            a_units="rad/s",
            b_units="m/s",
            vec_size=nn,
        )

        cross_prod_1.add_product(
            "w_b_cross_I_times_w_b",
            a_name="w_b",
            b_name="I_times_w_b",
            c_units="N*m",
            a_units="rad/s",
            b_units="kg*m**2/s",
            vec_size=nn,
        ),

        self.add_subsystem("mat_vec_prod_1", mat_vec_prod, promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("cross_prod_1", cross_prod_1, promotes_inputs=["*"], promotes_outputs=["*"])

        adder = om.AddSubtractComp()

        adder.add_equation(
            "V_b_rate",
            ["F_by_m", "w_b_cross_V_b"],
            vec_size=nn,
            length=3,
            val=0.0,
            units="m/s**2",
            scaling_factors=[1, -1],
        )
        adder.add_equation(
            "w_b_rate_eqn_LHS",
            ["M", "w_b_cross_I_times_w_b"],
            vec_size=nn,
            length=3,
            val=0.0,
            units="N*m",
            scaling_factors=[1, -1],
        )
        self.add_subsystem("adder_1", adder, promotes_inputs=["*"], promotes_outputs=["*"])

        lin_sys = om.LinearSystemComp(
            size=3,
            vec_size=nn,
            vectorize_A=True,
        )

        self.add_subsystem(
            "body_rates",
            lin_sys,
            promotes_inputs=[("A", "I"), ("b", "w_b_rate_eqn_LHS")],
            promotes_outputs=[("x", "w_b_rate")],
        )
        self.add_subsystem(
            "quaternion_rates",
            QuaternionRates(num_nodes=nn),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        self.set_input_defaults("I", units="kg*m**2", val=np.tile(np.eye(3), [nn, 1, 1]))
        self.set_input_defaults("F_by_m", units="m/s**2", val=np.zeros([nn, 3]))
        self.set_input_defaults("M", units="N*m", val=np.zeros([nn, 3]))


class QuaternionRates(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("R", val=np.ones([nn, 4]), desc="Rotation Quaternion")
        self.add_input("w_b", val=np.ones([nn, 3]), units="rad/s", desc="body angular rates")
        self.add_output("R_rate", val=np.zeros([nn, 4]), desc="Rotation Quaternion rates", units="1/s")

        # permutation matrices to get quaternion multiplication
        self.P = np.array(
            [
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]],
                [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]],
            ]
        )

        self.P_reduced = np.delete(self.P, 0, axis=2)

        range4 = np.arange(4)
        rows = []
        cols = []
        # set the jacobian sparsity pattern, iterante through num nodes first and fill out the diagonal
        for k in np.arange(nn):
            for i in range4:
                for j in range4:
                    rows += [4 * k + i]
                    cols += [4 * k + j]
        # print(rows)
        # print(cols)
        self.declare_partials("R_rate", "R", rows=rows, cols=cols)

        range3 = np.arange(3)
        rows = []
        cols = []
        for k in np.arange(nn):
            for i in range4:
                for j in range3:
                    rows += [4 * k + i]
                    cols += [3 * k + j]

        self.declare_partials("R_rate", "w_b", rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        nn = self.options["num_nodes"]
        R = inputs["R"]
        omega = inputs["w_b"]

        omega_quat = np.column_stack((np.zeros(nn), omega))

        # R_w, R_x, R_y, R_z = R[:, 0], R[:, 1], R[:, 2], R[:, 3]

        # om_x, om_y, om_z = omega[:, 0], omega[:, 1], omega[:, 2]

        # omega ordered
        # R_dot[:, 0] = R_w * om_w - R_x * om_x - R_y * om_y - R_z * om_z
        # R_dot[:, 1] = R_x * om_w + R_w * om_x - R_z * om_y + R_y * om_z
        # R_dot[:, 2] = R_y * om_w + R_z * om_x + R_w * om_y - R_x * om_z
        # R_dot[:, 3] = R_z * om_w - R_y * om_x + R_x * om_y + R_w * om_z

        # r ordered
        # R_dot[:, 0] = R_w * om_w - R_x * om_x - R_y * om_y - R_z * om_z
        # R_dot[:, 1] = R_w * om_x + R_x * om_w + R_y * om_z - R_z * om_y
        # R_dot[:, 2] = R_w * om_y - R_x * om_z + R_y * om_w + R_z * om_x
        # R_dot[:, 3] = R_w * om_z + R_x * om_y - R_y * om_x + R_z * om_w

        R_rate = 0.5 * np.einsum("ij,ljk,ik->il", R, self.P, omega_quat)

        outputs["R_rate"] = R_rate

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]
        R = inputs["R"]
        omega = inputs["w_b"]
        omega_quat = np.column_stack((np.zeros(nn), omega))

        # magic formulas. Taking partials involves not dropping the index over which you sum in the formula

        J["R_rate", "R"] = 0.5 * np.einsum("ljk,ik->ilj", self.P, omega_quat).flatten()

        J["R_rate", "w_b"] = 0.5 * np.einsum("ljk,ij->ilk", self.P_reduced, R).flatten()


class DownVector(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare(
            "p_origin", default=np.array([0, 0, 0]), desc="Origin of inertial frame. Can make it launch site"
        )

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("p", val=np.zeros([nn, 3]), units="m", desc="position in ECI frame")
        self.add_input("body_angles", val=np.zeros([nn, 3]), units="rad", desc="body angles")


class GravityForce(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare(
            "p_origin", default=np.array([0, 0, 0]), desc="Origin of inertial frame. Can make it launch site"
        )

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("p", val=np.zeros([nn, 3]), units="m", desc="position in ECI frame")
        self.add_input("body_angles", val=np.zeros([nn, 3]), units="rad", desc="body angles")
        self.add_output("F_grav", val=np.zeros([nn, 3]), units="N", desc="gravity force vector")


if __name__ == "__main__":
    p = om.Problem()
    p.model.add_subsystem("six_dof", SixDOFGroup(num_nodes=2))
    p.setup(force_alloc_complex=True)
    p.set_val("six_dof.V_b", val=np.array([1, 10, 100]), units="m/s")
    p.set_val("six_dof.w_b", val=np.array([2, 3, 4]), units="rad/s")
    p.set_val("six_dof.R", val=np.array([1, 2, 3, 4]))
    p.set_val("six_dof.F_by_m", val=np.array([10, 0, 0]), units="m/s**2")
    p.set_val("six_dof.M", val=np.array([1, -1, 1]), units="N*m")
    p.run_model()
    partials = p.check_partials(
        compact_print=True,
        method="fd",
    )
    om.n2(p)
