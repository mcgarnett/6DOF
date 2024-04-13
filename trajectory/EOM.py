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

        adder = om.AddSubtractComp(
            "V_b_rate",
            ["a_b", "w_b_cross_V_b"],
            vec_size=nn,
            length=3,
            val=0.0,
            units="m/s**2",
            scaling_factors=[1, -1],
        )

        adder.add_equation(
            "w_b_rate_eqn_LHS",
            ["M_total_b", "w_b_cross_I_times_w_b"],
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
        self.set_input_defaults("a_b", units="m/s**2", val=np.zeros([nn, 3]))
        self.set_input_defaults("M_total_b", units="N*m", val=np.zeros([nn, 3]))
        # self.set_input_defaults("R", val=np.ones([nn, 4]))
        # self.set_input_defaults("down_vec_i",val=np.tile())


class QuaternionGroup(om.Group):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        magnitude = om.VectorMagnitudeComp(vec_size=nn, length=4, in_name="R", mag_name="R_mag", units=None)
        self.add_subsystem("quat_magnitude", magnitude, promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("quat_normalize", NormalizeQuat(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("quat_inverse", InvertQuat(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])

        self.set_input_defaults("R", val=np.zeros([nn, 4]))


class FrameRotationsGroup(om.Group):
    def __init__(self, **kwargs):
        """
        Initialize the frame rotation component.
        """
        super().__init__(**kwargs)

        self._variables = []

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_subsystem(
            "rotation_matrix",
            RotationMatrix(num_nodes=nn),
            promotes_inputs=[("R", "R_normed")],
            promotes_outputs=["C"],
        )

        for i, (input_var, output_var, units) in enumerate(self._variables):
            if i == 0:
                mat_vec_prod_2 = om.MatrixVectorProductComp(
                    A_name="C",
                    A_units=None,
                    x_name=input_var,
                    x_units=units,
                    b_name=output_var,
                    b_units=units,
                    vec_size=nn,
                    A_shape=(3, 3),
                )

            else:
                mat_vec_prod_2.add_product(
                    A_name="C",
                    A_units=None,
                    x_name=input_var,
                    x_units=units,
                    b_name=output_var,
                    b_units=units,
                    vec_size=nn,
                    A_shape=(3, 3),
                )

        self.add_subsystem(
            "rotation",
            mat_vec_prod_2,
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

    def add_variable(self, var):
        self._variables += [var]


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


class RotationMatrix(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("R", val=np.ones([nn, 4]), desc="Rotation Quaternion")
        self.add_output("C", val=np.zeros([nn, 3, 3]), desc="vector to rotate")
        rows = []
        cols = []
        for k in np.arange(nn):
            for i in np.arange(9):
                for j in np.arange(4):
                    rows += [9 * k + i]
                    cols += [4 * k + j]

        self.declare_partials("C", "R", rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        R = inputs["R"]
        q0, q1, q2, q3 = R[:, 0], R[:, 1], R[:, 2], R[:, 3]

        c00 = q0**2 + q1**2 - q2**2 - q3**2
        c01 = 2 * (q1 * q2 - q0 * q3)
        c02 = 2 * (q1 * q3 + q0 * q2)
        c10 = 2 * (q1 * q2 + q0 * q3)
        c11 = q0**2 - q1**2 + q2**2 - q3**2
        c12 = 2 * (q2 * q3 - q0 * q1)
        c20 = 2 * (q1 * q3 - q0 * q2)
        c21 = 2 * (q2 * q3 + q0 * q1)
        c22 = q0**2 - q1**2 - q2**2 + q3**2
        # Rotation Matrix
        C = np.array([[c00, c01, c02], [c10, c11, c12], [c20, c21, c22]])
        C = np.moveaxis(C, 2, 0)
        outputs["C"] = C

    def compute_partials(self, inputs, J):
        R = inputs["R"]
        q0, q1, q2, q3 = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
        # C elements in rows of J, wrt q components in columns of J

        # c00 = q0**2 + q1**2 - q2**2 - q3**2
        j00 = 2 * q0
        j01 = 2 * q1
        j02 = -2 * q2
        j03 = -2 * q3

        # c01 = 2 * (q1 * q2 - q0 * q3)
        j10 = -2 * q3
        j11 = 2 * q2
        j12 = 2 * q1
        j13 = -2 * q0

        # c02 = 2 * (q1 * q3 + q0 * q2)
        j20 = 2 * q2
        j21 = 2 * q3
        j22 = 2 * q0
        j23 = 2 * q1

        # c10 = 2 * (q1 * q2 + q0 * q3)
        j30 = 2 * q3
        j31 = 2 * q2
        j32 = 2 * q1
        j33 = 2 * q0

        # c11 = q0**2 - q1**2 + q2**2 - q3**2

        j40 = 2 * q0
        j41 = -2 * q1
        j42 = 2 * q2
        j43 = -2 * q3

        # c12 = 2 * (q2 * q3 - q0 * q1)
        j50 = -2 * q1
        j51 = -2 * q0
        j52 = 2 * q3
        j53 = 2 * q2

        # c20 = 2 * (q1 * q3 - q0 * q2)
        j60 = -2 * q2
        j61 = 2 * q3
        j62 = -2 * q0
        j63 = 2 * q1

        # c21 = 2 * (q2 * q3 + q0 * q1)
        j70 = 2 * q1
        j71 = 2 * q0
        j72 = 2 * q3
        j73 = 2 * q2

        # c22 = q0**2 - q1**2 - q2**2 + q3**2
        j80 = 2 * q0
        j81 = -2 * q1
        j82 = -2 * q2
        j83 = 2 * q3

        J_mat = np.array(
            [
                [j00, j01, j02, j03],
                [j10, j11, j12, j13],
                [j20, j21, j22, j23],
                [j30, j31, j32, j33],
                [j40, j41, j42, j43],
                [j50, j51, j52, j53],
                [j60, j61, j62, j63],
                [j70, j71, j72, j73],
                [j80, j81, j82, j83],
            ]
        )
        J_mat = np.moveaxis(J_mat, 2, 0)
        # print(J_mat.shape)
        J["C", "R"] = J_mat.flatten()


class NormalizeQuat(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("R", val=np.zeros([nn, 4]), units=None)
        self.add_input("R_mag", val=np.zeros(nn), units=None)
        self.add_output("R_normed", val=np.zeros([nn, 4]), units=None)

        ar = np.arange(nn * 4)
        self.declare_partials("R_normed", "R", rows=ar, cols=ar)

        rows = []
        cols = []
        for k in np.arange(nn):
            for i in np.arange(4):
                rows += [4 * k + i]
                cols += [k]

        self.declare_partials("R_normed", "R_mag", rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        R = inputs["R"]
        R_mag = inputs["R_mag"]

        outputs["R_normed"] = np.einsum("ij, i -> ij", R, 1 / R_mag)

    def compute_partials(self, inputs, J):
        R = inputs["R"]
        R_mag = inputs["R_mag"]

        J["R_normed", "R_mag"] = np.einsum("ij, i -> ij", R, -1 / R_mag**2).flatten()
        J["R_normed", "R"] = np.repeat(1 / R_mag, 4)


class InvertQuat(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("R_normed", val=np.zeros([nn, 4]), units=None)
        self.add_output("R_inverse", val=np.zeros([nn, 4]), units=None)
        ar = np.arange(4 * nn)
        a = np.array([1, -1, -1, -1])
        self.a = a
        self.declare_partials("R_inverse", "R_normed", rows=ar, cols=ar, val=np.tile(a, nn))

    def compute(self, inputs, outputs):
        R = inputs["R_normed"]
        outputs["R_inverse"] = R * self.a


class Accel(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input("F_total_b", val=np.zeros([nn, 3]), units="N")
        self.add_input("g_b", val=np.zeros([nn, 3]), units="m/s**2")
        self.add_input("m", val=np.ones(nn), units="kg")
        self.add_output("a_b", val=np.ones([nn, 3]), units="m/s**2")

        num_jac_entries = 3 * nn

        rows = []
        cols = []

        for i in range(num_jac_entries):
            rows += [i]
            cols += [i // 3]

        self.declare_partials("a_b", "m", rows=rows, cols=cols)

        ar = np.arange(num_jac_entries)

        self.declare_partials("a_b", "F_total_b", rows=ar, cols=ar)
        self.declare_partials("a_b", "g_b", rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        F_b = inputs["F_total_b"]
        g_b = inputs["g_b"]
        m = inputs["m"]

        outputs["a_b"] = F_b / m[:, np.newaxis] + g_b

    def compute_partials(self, inputs, J):
        F = inputs["F_total_b"]
        m = inputs["m"]

        J["a_b", "F_total_b"] = np.repeat(1 / m, 3)
        J["a_b", "m"] = (-F / m[:, np.newaxis] ** 2).flatten()


class ForceAndMomentAdderGroup(om.Group):
    # TODO add dummy force so adder can deal with just one force
    def __init__(self, **kwargs):
        """
        Initialize the frame rotation component.
        """
        super().__init__(**kwargs)

        self._force_position_names = []
        self._moment_names = []

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        force_names, output_names = [], []
        for i, (pos_name, force_name) in enumerate(self._force_position_names):
            output_name = force_name + "_moment"
            force_names += [force_name]
            output_names += [output_name]

            if i == 0:
                cross_prod = om.CrossProductComp(
                    c_name=output_name, a_name=pos_name, b_name=force_name, c_units="N*m", a_units="m", b_units="N", vec_size=nn
                )
            else:
                cross_prod.add_product(
                    c_name=output_name, a_name=pos_name, b_name=force_name, c_units="N*m", a_units="m", b_units="N", vec_size=nn
                )

        adder = om.AddSubtractComp()

        adder.add_equation("F_total_b", force_names, vec_size=nn, length=3, val=0.0, units="N")

        adder.add_equation("M_total_b", output_names + self._moment_names, vec_size=nn, length=3, val=0.0, units="N*m")

        self.add_subsystem("cross_prod", cross_prod, promotes=["*"])

        self.add_subsystem("adder", adder, promotes=["*"])

        self.add_subsystem("accel", Accel(num_nodes=nn), promotes=["*"])

        for pos_name, force_name in self._force_position_names:
            self.set_input_defaults(pos_name, val=np.zeros([nn, 3]))

    def add_force(self, position_name, force_name):
        self._force_position_names += [(position_name, force_name)]

    def add_moment(self, moment_name):
        self._moment_names += moment_name


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
        method="cs",
    )
    om.n2(p)
