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
File Name: main.py
## =========================================================================== ## 
"""


# System (Default Lib.)
import sys
# Own library for robot control (kinematics), visualization, etc. (See manipulator.py)
import manipulator as m

# Test Results (Select one of the options FK, IK, BOTH)
test_kin = 'IK'

fk_test_pose = [0.0, 0.0, 0.0]
ik_test_pose = [200, 100]


def main():
    # Initial Parameters -> ABB IRB910SC
    # Product Manual: https://search.abb.com/library/Download.aspx?DocumentID=3HAC056431-001&LanguageCode=en&DocumentPartId=&Action=Launch

    # Working range (Axis 1, Axis 2, ...)
    axis_wr = [[-140.0, 140.0], [-150.0, 150.0], [-150.0, 150.0]]
    joints = [0.0, 0.0, 0.0]
    # Length of Arms (Link 1, Link2, ...)
    arm_length = [250.0, 200.0, 100.0]

    # DH (Denavit-Hartenberg) parameters

    # Initialization of the Class (Control Manipulator)
    # Input:
    #   (1) Robot name         [String]
    #   (2) Axis working range [Float Array]
    #   (3) Arm lengths        [Float Array]

    scara = m.ScaraControl('ABB IRB 910SC (SCARA)',
                           axis_wr, arm_length, joints)

    if test_kin == 'FK':
        scara.forward_kinematics(0, fk_test_pose, True)
    elif test_kin == 'IK':
        scara.inverse_kinematics(ik_test_pose, 0)
    elif test_kin == 'BOTH':
        scara.forward_kinematics(0, fk_test_pose, True)
        scara.inverse_kinematics(scara.ee_pose, 1)

    # 1. Display the entire environment with the robot and other functions.
    # 2. Display the work envelope (workspace) in the environment (depends on input).
    # Input:
    #  (1) Work Envelop Parameters
    #       a) Visible                   [BOOL]
    #       b) Type (0: Mesh, 1: Points) [INT]
    scara.display_environment([False, 0])


if __name__ == '__main__':
    sys.exit(main())
