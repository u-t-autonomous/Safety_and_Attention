#! /usr/bin/env/ python2

# Imports for ROS side
import rospy
import numpy as np
from numpy import linalg as LA
import sys
import types
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, PoseStamped
from std_msgs.msg import Bool
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import LaserScan, PointCloud2
from tf.transformations import euler_from_quaternion
from Safety_and_Attention.msg import Ready
import time
# Imports for Algorithm side
import copy
import random
from ViconTracker import Tracker
from TBVelocityController import TBVelocityController as controller
# from partial_semantics import *
# Rviz
# from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Quaternion, Vector3
import command_control as cc

# # To import for running safety and attention simulations
from linear_dynamics_auxilliary_functions import (ellipsoid_level_function,
    check_collisions,continue_condition,get_mu_sig_over_time_hor,
    get_mu_sig_theta_over_time_hor,get_Q_mats_over_time_hor,
    get_Qplus_mats_over_time_hor,propagate_linear_obstacles,
    propagate_nonlinear_obstacles)
from unicycle_dc_motion_planning_ecos import \
    (ecos_unicycle_shared_cons, solve_obs_free_ecos_unicycle,
     stack_params_for_ecos, dc_motion_planning_call_ecos_unicycle)
from unicycle_Robot_SaA_Environment import LinearObstacle, NonlinearObstacle, RobotSaAEnvironment
from ecosqp_file import ecosqp

# For solving MPC problem
import casadi as csi
import osqp
import ecos
from scipy.sparse import csc_matrix as csc

# FOR MAPPING USING THE VICON AND TURTLEBOT CAMERA SYSTEM
from detect_colors_hardware import object_map


# ------------------ Start Function Definitions ---------------
def make_user_wait(msg="Enter exit to exit"):
    data = raw_input(msg + "\n")
    if data == 'exit':
        sys.exit()

def wait_for_time():
    """Wait for simulated time to begin """
    while rospy.Time().now().to_sec() == 0:
        pass

def build_massage(data):
    vel_cmd = Twist()

    vel_cmd.linear.x = data[0]
    vel_cmd.linear.y = data[1]
    vel_cmd.linear.z = data[2]

    vel_cmd.angular.x = data[3]
    vel_cmd.angular.y = data[4]
    vel_cmd.angular.z = data[5]

    return vel_cmd
# ------------------ End Function Definitions -----------------

# Adjusted 'main' to run using unicycle dynamics
if __name__ == '__main__':

    np.random.seed(0)

    rospy.init_node("robot_control_0_node", anonymous=True)
    wait_for_time()

    vicon_track = Tracker()
    vicon_offset_x = -1.092
    vicon_offset_y = -1.533

    robot_name='tb3_0'

########  Start new stuff ########

    #-- This is the publisher object that you feed your velocity command to --#
    cmd_vel_pub = rospy.Publisher('/tb3_0/cmd_vel', Twist, queue_size = 1)

    #-- Build your message with velocity set point data --#
    data = None #<<<------- Fill this with your data
    vel_cmd = build_massage(data)

    #-- Publish the velocity command using the publisher object --#
    cmd_vel_pub.publish(vel_cmd)

    # # #-- This is the upateded controller. It will not perform a final rotation --# # #
    # # #-- Uncomment if you want to use it --# # #
    # vel_controller_0 = controller('/tb3_0/odom', '/tb3_0/cmd_vel',vicon_track,robot_name)
    # rospy.sleep(1.0)


########  Back to old stuff ########


    # Create the obstacle map
    ''' NOTE THAT:
    obstacle 1 maps to 'red'
    obstacle 2 maps to 'green'
    obstacle 3 maps to 'blue'
    FOR NOW, ASSUME THAT WE ARE ALWAYS USING 3 OBSTACLES! '''
    obstacle_tracker = object_map()

    ''' Set up the safety and attention environment to perform the planning in,
        here we are approximating unicycle dynamics'''

    # Set the initial point of the robotic agent in the Gazebo world (make sure this
    # is the same as the initial position in the Safety and Attention environment
    rob_heading_ang = 0
    # init_point_0 = [Point(-2.5, -0.75, None), rob_heading_ang]   # the list is [Point(x,y,z),heading angle
                                                                # (measured counter-clockwise from x-axis)]
    # vel_controller_0.go_to_point(init_point_0)

    # Observation strategy of the robot. Either "bin" or "sum" (sum is the preferred choice)
    rob_obs_strat = "sum"

    # Robotic agent follows linear dynamics x_k+1 = A*x_k + B*u_k
    rob_A_mat = np.eye(2)

    # Field of view of robot, assumed symmetric about current heading angle
    max_heading_view = 0.541

    # Min and max velocity input
    rob_min_velocity = 0.01
    rob_max_velocity = 0.25

    # Maximum change in angle in one time step (again, assume symmetric about
    # the current heading angle)
    rob_max_turn_rate = np.pi/3

    # One-step "aggressive" turn to view most relevant obstacle
    rob_agg_turn_rate = np.pi/3

    # Current most relevant obstacle
    most_rel_obs_ind = None

    # Number of turning rates to consider and the associated array of turning rates
    num_turning_rates = 61
    turning_rates_array = np.linspace(-rob_max_turn_rate,rob_max_turn_rate,num_turning_rates)

    # Define the state that the robotic agent must travel to and the L2-norm
    # tolerance for which we can say that the agent "reached" that state
    goal_state = np.array([[1.9], [1.9]])
    goal_tolerance = 10e-2

    # Parameter for constructing Gaussian level sets of obstacle positions
    # A lower beta means that the "keep-out region" is larger
    beta = 0.01

    # Number of time steps into the future that the robotic agent must plan
    planning_horizon = 20

    # Initial position of the robotic agent in the environment
    rob_init_pos = np.array([[-2.5],[-0.75]])

    # The size of the circle (assumed to be in meters?) for which the robotic
    # agent can make an observation about an obstacle if the obstacle is within
    # that position.
    obs_field_of_view_rad = 100

    # The number of time steps between subsequent opportunities for the robotic
    # agent to make an observation
    obs_interval = 1

    # Assuming a square, the absolute value in the L1 sense that the position of
    # the robotic agent can be in any of the cardinal directions
    rob_state_x_max = 3.0
    rob_state_y_max = 2.0

    # Time between subsequent time steps "k" and "k+1" (again, not really a physical
    # parameter, will probably need to play around with this value)
    sampling_time = 0.25

    # Parameters weighting how to weight turning towards obstacle vs reaching goal state
    discount_factor = 0.7
    discount_weight = 10

    # Now that we have all of the ingredients, create the robot safety and
    # attention environment
    robotic_agent_environ = RobotSaAEnvironment(goal_state, goal_tolerance, beta,
                                                planning_horizon, rob_init_pos, rob_A_mat,
                                                obs_field_of_view_rad, obs_interval, rob_state_x_max,
                                                rob_state_y_max, sampling_time, rob_obs_strat,
                                                max_heading_view, rob_min_velocity, rob_max_velocity,
                                                rob_max_turn_rate, rob_agg_turn_rate, most_rel_obs_ind,
                                                num_turning_rates, turning_rates_array, rob_heading_ang,
                                                discount_factor, discount_weight)

    # Add the first obstacle
    # obs_1_init = np.array([-1.4, -0.3])
    obs_1_init = np.array([[-0.75], [-2.00]])
    obs_1_A_matrix = np.eye(2)
    obs_1_F_matrix = np.eye(2)
    obs_1_mean_vec = np.array([[0.0],[0.2]])
    obs_1_cov_mat = np.array([[0.008, 0.001], [0.001, 0.008]])
    obs_1_radius = 0.25
    robotic_agent_environ.add_linear_obstacle(obs_1_init, obs_1_A_matrix,
                                              obs_1_F_matrix, obs_1_mean_vec, obs_1_cov_mat, obs_1_radius)

    # Add the second obstacle
    # obs_2_init = np.array([1.3, 1.])
    obs_2_init = np.array([[1.2], [1.8]])
    obs_2_A_matrix = np.eye(2)
    obs_2_F_matrix = np.eye(2)
    obs_2_mean_vec = np.array([[0.05],[-0.1]])
    obs_2_cov_mat = np.array([[0.004, 0.0015], [0.0015, 0.005]])
    obs_2_radius = 0.25
    robotic_agent_environ.add_linear_obstacle(obs_2_init, obs_2_A_matrix,
                                              obs_2_F_matrix, obs_2_mean_vec, obs_2_cov_mat, obs_2_radius)

    # Add the third obstacle
    # obs_3_init = np.array([-0.3, -1.2])
    obs_3_init = np.array([[2.25], [-2.0]])
    obs_3_A_matrix = np.eye(2)
    obs_3_F_matrix = np.eye(2)
    obs_3_mean_vec = np.array([[0.025],[0.15]])
    obs_3_cov_mat = np.array([[0.005, 0.0015], [0.0015, 0.008]])
    obs_3_radius = 0.25
    robotic_agent_environ.add_linear_obstacle(obs_3_init, obs_3_A_matrix,
                                              obs_3_F_matrix, obs_3_mean_vec, obs_3_cov_mat, obs_3_radius)

    # Construct the solvers and get the bounds.
    robotic_agent_environ.construct_mpc_solvers_and_get_bounds()

    # Wait until all other robots are ready
    rdy = cc.ReadyTool(robot_name)
    print("*** Robot {} is ready and waiting to start ***".format(int(robot_name[-1])))
    rdy.set_ready(True)
    # rdy.wait_for_ready()
    # print("Robot {} made it past Ready Check *".format(int(robot_name[-1]))) # Comment when done testing
    # sys.exit() # Comment when done testing

    # Now, while we have not reached the target point, continue executing the controller
    solve_optimization_times = []
    travel_to_point_times = []
    while robotic_agent_environ.continue_condition:

        # Resolve the problem and extract the next state to transition to, pass
        # this value to the velocity controller
        tic_sop_1 = time.time()
        robotic_agent_environ.solve_optim_prob_and_update()
        solve_optimization_times.append(time.time() - tic_sop_1)
        print(time.time() - tic_sop_1)

        # Assume that the nominal trajectory also has the heading angle stacked underneath,
        # making the matrix exist in R^((state_dim+1)x(time_horizon))
        next_p = robotic_agent_environ.nominal_trajectory[:, 0]
        next_yaw = robotic_agent_environ.heading_angle_sequence[0]

        # Need to convert yaw in [0,2pi] to [-pi,pi]
        if 0<=next_yaw <= np.pi:
            next_yaw = next_yaw
        elif np.pi < next_yaw <= 2*np.pi:
            next_yaw = next_yaw - 2*np.pi
        elif 0 > next_yaw >= -np.pi:
            next_yaw = next_yaw
        else:  # Must be that the yaw angle is between -pi and -2*pi -> want between 0 and pi
            next_yaw = next_yaw + 2*np.pi

        next_point = Point(float(next_p[0]) - (-2.50), float(next_p[1]) - (-0.75), None)
        next_state = [next_point,next_yaw]

        rdy.set_ready(False)
        tic_gtp_1 = time.time()
        vel_controller_0.go_to_point(next_state)
        travel_to_point_times.append(time.time() - tic_gtp_1)
        rdy.set_ready(True)
        # Wait for the agent and the obstacles to have synchronized to their next state
        # rdy.wait_for_ready()
        print("Robot {} is moving to the next waypoint *".format(int(robot_name[-1])))

        # Query the current position of each obstacle
        obs_1_x = vicon_track.data[3].translation.x - vicon_offset_x
        obs_1_y = vicon_track.data[3].translation.y - vicon_offset_y
        obs_2_x = vicon_track.data[2].translation.x - vicon_offset_x
        obs_2_y = vicon_track.data[2].translation.y - vicon_offset_y
        obs_3_x = vicon_track.data[1].translation.x - vicon_offset_x
        obs_3_y = vicon_track.data[1].translation.y - vicon_offset_y
        obs_1_cur_loc = np.array([[obs_1_x], [obs_1_y]])
        obs_2_cur_loc = np.array([[obs_2_x], [obs_2_y]])
        obs_3_cur_loc = np.array([[obs_3_x], [obs_3_y]])
        obs_act_positions = [obs_1_cur_loc, obs_2_cur_loc, obs_3_cur_loc]

        # Query camera to determine which obstacles are in view
        obs_in_view = obstacle_tracker.which_obj()

        # Convert obs in view from color to number of obtacle
        obs_in_view_list = []
        if 'pink' in obs_in_view:
            obs_in_view_list.append(0)
        if 'green' in obs_in_view:
            obs_in_view_list.append(1)
        if 'yellow' in obs_in_view:
            obs_in_view_list.append(2)

        # Now, update the simulated and actual positions of the robot, obstacles.
        robotic_agent_environ.update_obs_positions_and_plots(obs_act_positions,obs_in_view_list)
        robotic_agent_environ.rob_pos = np.array([vicon_track.data[0].translation.x - vicon_offset_x, vicon_track.data[0].translation.y - vicon_offset_y])
        rot_q = vicon_track.data[0].rotation
        _, _, vicon_yaw = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        robotic_agent_environ.heading_angle = vicon_yaw
        # robotic_agent_environ.rob_pos = np.array([vel_controller_0.x, vel_controller_0.y])
        # robotic_agent_environ.heading_angle = vel_controller_0.yaw

        # print('----------')
        # print(robotic_agent_environ.most_rel_obs_ind)
        # print(robotic_agent_environ.heading_angle)
        # print(robotic_agent_environ.best_gamma_ind)
        # print(robotic_agent_environ.heading_angle_sequence)

    np.savetxt("optimization_times_sum.csv", np.array(solve_optimization_times),delimiter=',')
    np.savetxt("go_to_point_times.csv", np.array(travel_to_point_times),delimiter=',')