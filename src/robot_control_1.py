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

    rospy.init_node("robot_control_1_node", anonymous=True)
    wait_for_time()

    # Create velocity controller and converter objects
    vel_controller_1 = controller('/tb3_0/odom', '/tb3_0/cmd_vel')
    robot_name='tb3_0'
    rospy.sleep(1.0)

    # Set the initial point of the robotic agent in the Gazebo world (make sure this
    # is the same as the initial position in the Safety and Attention environment)
    vel_controller_1.go_to_point(Point(-1, -1, None))

    # rospy.sleep(1.0)

    vel_controller_1.go_to_point(Point(1, -2, None))


    # rospy.sleep(1.0)

    vel_controller_1.go_to_point(Point(-1, -1, None))

    # # Dynamics of first obstacle
    # sampling_time = 1.
    # obs_1_A_matrix = np.eye(2)
    # obs_1_F_matrix = sampling_time*np.eye(2)
    # obs_1_mean_vec = np.array([0.20, 0.0])
    # obs_1_cov_mat = np.array([[0.0075, 0.001], [0.001, 0.0075]])

    # # Generate a set of waypoints for the first obstacle to follow
    # num_steps = 200
    # traj = []
    # traj_np = []
    # for step in range(0,num_steps):
    #     if step == 0:
    #         # Query the previous point in the set of waypoints
    #         prev_x = init_point_1.x
    #         prev_y = init_point_1.y
    #         prev_state = np.array([prev_x,prev_y])
    #         # Push this point through the dynamics
    #         obs_w_step = np.random.multivariate_normal(obs_1_mean_vec,obs_1_cov_mat,1)
    #         obs_w_step = np.reshape(obs_w_step, (2,))
    #         new_state = np.matmul(obs_1_A_matrix,prev_state) + np.matmul(obs_1_F_matrix,obs_w_step)
    #         new_point = Point(float(new_state[0]),float(new_state[1]), None)
    #         traj = [new_point]
    #         traj_np = [np.array(new_state)]
    #     else:
    #         # Query the previous point in the set of waypoints
    #         prev_x = traj[step-1].x
    #         prev_y = traj[step-1].y
    #         prev_state = np.array([prev_x, prev_y])
    #         # Push this point through the dynamics
    #         obs_w_step = np.random.multivariate_normal(obs_1_mean_vec, obs_1_cov_mat, 1)
    #         obs_w_step = np.reshape(obs_w_step, (2,))
    #         new_state = np.matmul(obs_1_A_matrix, prev_state) + np.matmul(obs_1_F_matrix, obs_w_step)
    #         new_point = Point(float(new_state[0]), float(new_state[1]), None)
    #         traj.append(new_point)
    #         traj_np.append(np.array(new_state))
    # #
    # np.save("obstacle_1_trajectory",np.array(traj_np))

    # # Wait until all other robots are ready
    # rdy = cc.ReadyTool(robot_name)
    # print("*** Robot {} is ready and waiting to start ***".format(int(robot_name[-1])))
    # rdy.set_ready(True)
    # rdy.wait_for_ready()
    # # print("Robot {} made it past Ready Check *".format(int(robot_name[-1]))) # Comment when done testing
    # # sys.exit() # Comment when done testing

    # #traj = np.load('obs_1_states.npy')

    # # Now, while we have not reached the target point, continue executing the controller
    # while not rospy.is_shutdown():
    #     for next_p in traj_np:
    #         rdy.set_ready(False)
    #         next_x = next_p[0]
    #         next_y = next_p[1]
    #         next_point = Point(next_x,next_y,None)
    #         vel_controller_1.go_to_point(next_point)
    #         rdy.set_ready(True)
    #         # Wait for the agent and the obstacles to have synchronized to their next state
    #         rdy.wait_for_ready()
    #         print("Robot {} is moving to the next waypoint *".format(int(robot_name[-1])))

