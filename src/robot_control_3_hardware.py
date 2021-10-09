#! /usr/bin/env/ python2

# Imports for ROS side
import rospy
import numpy as np
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from tf.transformations import euler_from_quaternion
from Safety_and_Attention.msg import Ready
import command_control as cc
# Imports for Algorithm side
import random
from TBVelocityController import TBVelocityController as controller


# To import for running safety and attention simulations
# from linear_dynamics_auxilliary_functions import (ellipsoid_level_function,
#     check_collisions,continue_condition,get_mu_sig_over_time_hor,
#     get_mu_sig_theta_over_time_hor,get_Q_mats_over_time_hor,
#     get_Qplus_mats_over_time_hor,propagate_linear_obstacles,
#     propagate_nonlinear_obstacles)
# from dc_based_motion_planner import (create_dc_motion_planner,
#     solve_dc_motion_planning,create_cvxpy_motion_planner_obstacle_free)
# from dc_motion_planning_ecos import (dc_motion_planning_call_ecos,
#     stack_params_for_ecos,set_up_independent_ecos_params)
# from Robot_SaA_Environment import LinearObstacle, NonlinearObstacle, RobotSaAEnvironment


# ------------------ Start Function Definitions ---------------
def make_user_wait(msg="Enter exit to exit"):
    data = raw_input(msg + "\n")
    if data == 'exit':
        sys.exit()

def wait_for_time():
    """Wait for simulated time to begin """
    while rospy.Time().now().to_sec() == 0:
        pass
# ------------------ End Function Definitions -----------------

if __name__ == '__main__':

    np.random.seed(0)

    rospy.init_node("robot_control_3_node", anonymous=True)
    wait_for_time()

    # Create velocity controller and converter objects
    vel_controller_3 = controller('/tb3_3/odom', '/tb3_3/cmd_vel')
    robot_name='tb3_3'
    rospy.sleep(1.0)

    # Wait until all other robots are ready
    rdy = cc.ReadyTool(robot_name)
    print("*** Robot {} is ready and waiting to start ***".format(int(robot_name[-1])))
    rdy.set_ready(True)
    rdy.wait_for_ready()
    # print("Robot {} made it past Ready Check *".format(int(robot_name[-1]))) # Comment when done testing
    # sys.exit() # Comment when done testing

    # Set the initial point of the robotic agent in the Gazebo world (make sure this
    # is the same as the initial position in the Safety and Attention environment)
    # init_point_3 = Point(2.25, -2.0, None)
    init_point_3 = Point(0.0, 0.0, None)
    vel_controller_3.go_to_point(init_point_3)

    traj_np = np.load('obstacle_3_trajectory.npy')
    traj_np = traj_np - np.tile(np.array([[2.25,-2.00]]),(200,1))

    # Now, while we have not reached the target point, continue executing the controller
    while not rospy.is_shutdown():
        for next_p in traj_np:
            rdy.set_ready(False)
            next_x = next_p[0]
            next_y = next_p[1]
            next_point = Point(next_x, next_y, None)
            vel_controller_3.go_to_point(next_point)
            rdy.set_ready(True)
            # Wait for the agent and the obstacles to have synchronized to their next state
            rdy.wait_for_ready()
            print("Robot {} is moving to the next waypoint *".format(int(robot_name[-1])))

