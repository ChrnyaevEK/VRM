"""
## =========================================================================== ##
MIT License
Copyright (c) 2020 Roman Parak
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## =========================================================================== ##
Author   : Roman Parak
Email    : Roman.Parak@outlook.com
Github   : https://github.com/rparak
File Name: manipulator.py
## =========================================================================== ##
"""

# Numpy (Array computing Lib.) [pip3 install numpy]
import numpy as np
# Mtaplotlib (Visualization Lib.) [pip3 install matplotlib]
import matplotlib.pyplot as plt


class DHParameters:
    # << DH (Denavit-Hartenberg) parameters structure >> #
    def __init__(self, theta, a, d, alpha):
        # Angle about previous z, from old x to new x
        # Unit [radian]
        self.theta = theta
        # Length of the common normal. Assuming a revolute joint, this is the radius about previous z
        # Unit [metres]
        self.a = a
        # Offset along previous z to the common normal
        # Unit [metres]
        self.d = d
        # Angle about common normal, from old z axis to new z axis
        # Unit [radian]
        self.alpha = alpha


class BaseControl:
    def __init__(self, robot_name, ax_wr, arm_ln, joints):
        # Robot Name -> Does not affect functionality (only for user)
        self.robot_name = robot_name
        # Axis working range
        self.ax_wr = ax_wr
        # Axis lengths
        self.arm_ln = arm_ln

        # Current EE pose -> pose(x, y)
        self.pose = np.zeros(2)
        # Current joints -> joints(theta_1, theta_2, ... N joints)
        self.joints = joints  # Copy initial angles

        # Rounding index of calculation (accuracy)
        self.rounding_index = 10


class CalcControl(BaseControl):
    # Override
    def hook_fast_calc_fk(self, target_joints):
        """
        Description:
            Fast calculation of forward kinematics (in this case it is recommended to use).
            Method should return new pose.
        """

        raise NotImplementedError('Fast FK not implemented')

    # Override
    def hook_dh_calc_fk(self, rDH_param):
        """
        Description:
            Slower calculation of Forward Kinematics using the Denavit-Hartenberg parameter table.
            Method should return new pose. Use separate_translation_part to get resulting pose from
            transition matrix.
        """
        raise NotImplementedError('DH FK not implemented')

    # Override
    def hook_calc_ik(self, target_pose):
        """
        Description:
            Calculation of inverse kinematic. Method should return new joints.
        """
        raise NotImplementedError('IK not implemented')

    def calc_dh_matrix(self, index, rDH_param):
        """
        Description:
            Calculation of Denavit-Hartenberg parameter table.
        Args:
            (1) index [INT]: Index of episode (Number of episodes is depends on number of joints)
        Returns:
            (2) Ai_aux [Float Matrix 4x4]: Transformation Matrix in the current episode
        """
        i = index

        # Reset/Initialize matrix
        Ai_aux = np.array(np.identity(4), copy=False)

        # << Calulation First Row >>
        # Rotational Part
        Ai_aux[0, 0] = np.cos(rDH_param.theta[i])
        Ai_aux[0, 1] = -np.sin(rDH_param.theta[i]) * np.cos(rDH_param.alpha[i])
        Ai_aux[0, 2] = np.sin(rDH_param.theta[i]) * np.sin(rDH_param.alpha[i])

        # Translation Part
        Ai_aux[0, 3] = rDH_param.a[i] * np.cos(rDH_param.theta[i])

        # << Calulation Second Row >>
        # Rotational Part
        Ai_aux[1, 0] = np.sin(rDH_param.theta[i])
        Ai_aux[1, 1] = np.cos(rDH_param.theta[i]) * np.cos(rDH_param.alpha[i])
        Ai_aux[1, 2] = -np.cos(rDH_param.theta[i]) * np.sin(rDH_param.alpha[i])
        # Translation Part
        Ai_aux[1, 3] = rDH_param.a[i] * np.sin(rDH_param.theta[i])

        # << Calulation Third Row >>
        # Rotational Part
        Ai_aux[2, 0] = 0
        Ai_aux[2, 1] = np.sin(rDH_param.alpha[i])
        Ai_aux[2, 2] = np.cos(rDH_param.alpha[i])
        # Translation Part
        Ai_aux[2, 3] = rDH_param.d[i]

        # << Set Fourth Row >>
        # Rotational Part
        Ai_aux[3, 0] = 0
        Ai_aux[3, 1] = 0
        Ai_aux[3, 2] = 0
        # Translation Part
        Ai_aux[3, 3] = 1

        return Ai_aux

    def calc_dh_param(self, joints):
        return DHParameters(theta=joints,
                            a=self.arm_ln,
                            d=[0.0, 0.0, 0.0],
                            alpha=[0.0, 0.0, 0.0])

    def convert_deg_rad(self, joints_deg):
        return [j * (np.pi/180) for j in joints_deg]

    def separate_translation_part(self, T):
        """
        Description:
            Separation translation part from the resulting transformation matrix.
        """
        return [
            round(T[0, 3], self.rounding_index),
            round(T[1, 3], self.rounding_index),
        ]

    def forward_kinematics(self, calc_type, target_joints, degree_repr):
        """
        Description:
            Forward kinematics refers to the use of the kinematic equations of a robot to compute 
            the position of the end-effector from specified values for the joint parameters.
            Joint Angles (Theta_1, Theta_2) <-> Position of End-Effector (x, y)
        Args:
            (1) calc_type [INT]: Select the type of calculation (0: DH Table, 1: Fast).
            (2) joints [Float Array]: Joint angle of target in degrees.
            (3) degree_repr [BOOL]: Representation of the input joint angle (Degree).

        Examples:
            self.forward_kinematics(0, [0.0, 45.0])
        """

        # Test joint limits
        for j, (mn, mx) in zip(target_joints, self.ax_wr):
            assert mn < j < mx

        if degree_repr:
            target_joints = self.convert_deg_rad(target_joints)

        rDH_param = self.calc_dh_param(target_joints)
        self.joints = target_joints

        if calc_type == 0:
            self.pose = self.hook_fast_calc_fk(target_joints)
        if calc_type == 1:
            self.pose = self.hook_dh_calc_fk(rDH_param)

    def inverse_kinematics(self, pose, cfg):
        """
        Description:
            Inverse kinematics is the mathematical process of calculating the variable 
            joint parameters needed to place the end of a kinematic chain.
            Position of End-Effector (x, y) <-> Joint Angles (Theta_1, Theta_2)
        Args:
            (1) pose [Float Array]: Position (x, y) of the target in meters.
            (2) cfg [INT]: Robot configuration (IK Multiple Solutions).

        Examples:
            self.inverse_kinematics([0.45, 0.10], 0)
        """

        self.joints = self.hook_calc_ik(np.array(pose))

        # Calculate the forward kinematics from the results of the inverse kinematics.
        self.forward_kinematics(0, self.joints, False)


class VisControl(BaseControl):

    def display_workspace(self, display_type=0):
        """
        Description:
            Display the work envelope (workspace) in the environment.

        Args:
            (1) display_type [INT]: Work envelope visualization options (0: Mesh, 1: Points).

        Examples:
            self.display_workspace(0)
        """

        raise NotImplementedError('Not implemented for 3 joints')

        # Generate linearly spaced vectors for the each of joints.
        theta_1 = np.linspace(
            (self.ax_wr[0][0]) * (np.pi/180), (self.ax_wr[0][1]) * (np.pi/180), 100)
        theta_2 = np.linspace(
            (self.ax_wr[1][0]) * (np.pi/180), (self.ax_wr[1][1]) * (np.pi/180), 100)

        # Return coordinate matrices from coordinate vectors.
        [theta_1_mg, theta_2_mg] = np.meshgrid(theta_1, theta_2)

        # Find the points x, y in the workspace using the equations FK.
        x_p = (self.arm_ln[0]*np.cos(theta_1_mg) +
               self.arm_ln[1]*np.cos(theta_1_mg + theta_2_mg))
        y_p = (self.arm_ln[0]*np.sin(theta_1_mg) +
               self.arm_ln[1]*np.sin(theta_1_mg + theta_2_mg))

        if display_type == 0:
            plt.fill(x_p, y_p, 'o', c=[0, 1, 0, 0.05])
            plt.plot(x_p[0][0], y_p[0][0], '.',
                     label=u"Work Envelop", c=[0, 1, 0, 0.5])
        elif display_type == 1:
            plt.plot(x_p, y_p, 'o', c=[0, 1, 0, 0.1])
            plt.plot(x_p[0][0], y_p[0][0], '.',
                     label=u"Work Envelop", c=[0, 1, 0, 0.5])

    def display_environment(self, work_envelope=[False, 0], target_pose=None):
        """
        Description:
            Display the entire environment with the robot and other functions.

        Args:
            (1) work_envelope [Array (BOOL, INT)]: Work Envelop options (Visibility, Type of visualization (0: Mesh, 1: Points)).

        Examples:
            self.display_environment([True, 0], [...])
        """

        # Display FK/IK calculation results (depens on type of options)
        self.display_result()

        # Condition for visible work envelop
        if work_envelope[0] == True:
            self.display_workspace(work_envelope[1])

        if target_pose is not None:
            if (target_pose[0] != self.pose[0]) and (target_pose[1] != self.pose[1]):
                mfc = [1, 0, 0]
            else:
                mfc = [1, 1, 0]

            plt.plot(
                target_pose[0],
                target_pose[1],
                label=r'Target Position: $p_{(x, y)}$',
                marker='o',
                ms=30,
                mfc=mfc,
                markeredgecolor=[0, 0, 0],
                mew=5)

        # Init x,y for O (origin) and T (target) of link.
        xO, yO = 0, 0
        xT, yT = 0, 0

        abs_th = 0
        for i in range(len(self.arm_ln)):
            abs_th += self.joints[i]

            # Move origin to link end
            xO = xT
            yO = yT

            xT += self.arm_ln[i] * np.cos(abs_th)
            yT += self.arm_ln[i] * np.sin(abs_th)

            # Link
            plt.plot([xO, xT], [yO, yT], 'k:', linewidth=2)

            # Joint
            plt.plot(xO, yO, marker='o', ms=5, color='b',
                     label=f'Joint {i+1}: {self.ax_wr[i][0]}min, {self.ax_wr[i][1]}max')

        # EE as is
        plt.plot(self.pose[0], self.pose[1], marker='o',
                 ms=5, label=f'EE', color='g')

        lim = np.sum(self.arm_ln)

        # Set minimum / maximum environment limits
        plt.axis([-lim - 20, lim + 20, -lim - 20, lim + 20])

        # Set additional parameters for successful display of the robot environment
        plt.grid()
        plt.xlabel('x position [m]', fontsize=20, fontweight='normal')
        plt.ylabel('y position [m]', fontsize=20, fontweight='normal')
        plt.title(self.robot_name, fontsize=50, fontweight='normal')
        plt.legend(loc=0, fontsize=20)
        plt.show()

    def display_rDHp(self, rDH_param):
        """
        Description: 
            Display the DH robot parameters.
        """

        print(
            f'[INFO] The Denavit-Hartenberg modified parameters of robot {self.robot_name}:')
        print(
            f'[INFO] theta = [{", ".join(format(x, "10.5f") for x in rDH_param.theta)}]')
        print(
            f'[INFO] a     = [{", ".join(format(x, "10.5f") for x in rDH_param.a)}]')
        print(
            f'[INFO] d     = [{", ".join(format(x, "10.5f") for x in rDH_param.d)}]')
        print(
            f'[INFO] alpha = [{", ".join(format(x, "10.5f") for x in rDH_param.alpha)}]')

    def display_result(self, target_pose=None, target_joints=None):
        """
        Description: 
            Display of the result of the kinematics forward/inverse of the robot and other parameters.
        """

        print('[INFO] Result of Kinematics calculation:')
        print(f'[INFO] Robot: {self.robot_name}')

        if target_pose is not None:
            print('[INFO] Target Position (End-Effector):')
            print(
                f'[INFO] p_t  = [{", ".join(format(x, "10.5f") for x in  target_pose)}]')

        if target_joints is not None:
            print('[INFO] Target Position (Joint):')
            print(
                f'[INFO] Theta  = [{", ".join(format(x, "10.5f") for x in target_joints)}]')

        print('[INFO] Actual Position (End-Effector):')
        print(f'[INFO] p_ee = [x: {self.pose[0]:.5f}, y: {self.pose[1]:.5f}]')

        print('[INFO] Actual Position (Joint):')
        print(
            f'[INFO] Theta  = [{", ".join(format(x, "10.5f") for x in self.joints)}]')


class ScaraControl(CalcControl, VisControl):
    def hook_fast_calc_fk(self, target_joints):
        x, y = 0, 0

        abs_th = 0
        for arm_ln, rel_th in zip(self.arm_ln, target_joints):
            abs_th += rel_th

            x += np.cos(abs_th) * arm_ln
            y += np.sin(abs_th) * arm_ln

        return x, y

    def hook_dh_calc_fk(self, rDH_param):
        self.display_rDHp(rDH_param)

        T = np.array(np.identity(4))
        for i in range(len(self.arm_ln)):
            T = T @ self.calc_dh_matrix(i, rDH_param)

        return self.separate_translation_part(T)

    def hook_calc_ik(self, target_pose):
        # Inspired by:
        # http://rodolphe-vaillant.fr/entry/114/cyclic-coordonate-descent-inverse-kynematic-ccd-ik
        # https://github.com/MarijnStam/ik-ccd/blob/master/ccd.py
        # https://github.com/ekorudiawan/CCD-Inverse-Kinematics-2D/blob/master/sources/CCD-Inverse-Kinematics-2D.py

        iteration_limit = 5000  # Maximum iterations of CCD allowed
        threshold = 0.5

        tmp_joints = [*self.joints]

        for _ in range(iteration_limit):
            joint_poses = self.calc_joint_poses(tmp_joints)

            for j in range(len(self.joints) - 1, -1, -1):

                end_to_target = target_pose - joint_poses[-1]
                error = np.sqrt(end_to_target[0] ** 2 + end_to_target[1] ** 2)

                if error < threshold:
                    rel_joints = []  # abs joints to relative
                    last = 0
                    for i, tmp_joint in enumerate(tmp_joints):
                        rel_joints.append(tmp_joint - last)
                        last = tmp_joint

                    return rel_joints
                else:
                    cur_to_end = joint_poses[-1] - joint_poses[j]
                    cur_to_end_mag = np.sqrt(
                        cur_to_end[0] ** 2 + cur_to_end[1] ** 2)

                    cur_to_target = target_pose - joint_poses[j]
                    cur_to_target_mag = np.sqrt(
                        cur_to_target[0] ** 2 + cur_to_target[1] ** 2)

                    end_target_mag = cur_to_end_mag * cur_to_target_mag

                    if end_target_mag <= 0.0001:
                        cos_rot_ang = 1
                        sin_rot_ang = 0
                    else:
                        cos_rot_ang = (
                            cur_to_end[0] * cur_to_target[0] + cur_to_end[1] * cur_to_target[1]) / end_target_mag
                        sin_rot_ang = (
                            cur_to_end[0] * cur_to_target[1] - cur_to_end[1] * cur_to_target[0]) / end_target_mag

                    rot_ang = np.arccos(max(-1, min(1, cos_rot_ang)))

                    if sin_rot_ang < 0.0:
                        rot_ang = -rot_ang

                    tmp_joints[j] = tmp_joints[j] + rot_ang
        else:
            raise RuntimeError(
                "Could not reach target point within max iterations")

    def calc_joint_poses(self, joints):
        poses = [np.zeros(2)]

        for i in range(len(joints)):
            poses.append(np.array([
                poses[i][0] + np.cos(joints[i]) * self.arm_ln[i],
                poses[i][1] + np.sin(joints[i]) * self.arm_ln[i],
            ]))

        return poses
