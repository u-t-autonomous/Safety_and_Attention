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
<<<<<<< Updated upstream
from ecosqp_file import ecosqp
=======

# For solving MPC problem
import casadi as csi
import osqp
import ecos
from scipy.sparse import csc_matrix as csc
>>>>>>> Stashed changes

# FOR MAPPING USING THE VICON AND TURTLEBOT CAMERA SYSTEM
from detect_colors import object_map

# Parallel processing
from itertools import repeat
import multiprocessing.dummy as multiprocessing
from multiprocessing.pool import ThreadPool
from multiprocessing import freeze_support

# ------------------ Start Class Definitions ------------------

class VelocityController:
    """Simple velocity controller meant to be used with turtlebot3"""
    def __init__(self, odom_topic_name, cmd_vel_topic_name, debug=False):
        self.debug = debug
        self.__odom_sub = rospy.Subscriber(odom_topic_name, Odometry, self.__odomCB)
        self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic_name, Twist, queue_size = 1)

        self.x = None
        self.y = None
        self.yaw = None
        self.r = rospy.Rate(4)
        self.vel_cmd = Twist()

    def __odomCB(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

    def go_to_point(self, data):

        # input variable goal should be of type [geometry_msgs/Point,yaw]
        goal = data[0]
        final_yaw = data[1]

        print("Starting to head towards the waypoint")

        ''' First do the rotation towards the goal '''
        error_x = goal.x - self.x
        error_y = goal.y - self.y
        angle_to_goal = np.arctan2(error_y, error_x)
        angle_error = self.yaw - angle_to_goal

        if self.debug:
            print("Starting to rotate towards waypoint")

        while abs(angle_error) > 0.01:
            ''' Only useful if there is some slip/slide of the turtlebot while rotating '''
            # error_x = goal.x - self.x
            # error_y = goal.y - self.y
            # angle_to_goal = np.arctan2(error_y, error_x) # # #
            angle_error = self.yaw - angle_to_goal
            if self.debug:
                print("Angle to goal: {:.5f},   Yaw: {:.5f},   Angle error: {:.5f}".format(angle_to_goal, self.yaw, angle_error))
            if angle_to_goal >= 0:
                if self.yaw <= angle_to_goal and self.yaw >= angle_to_goal - np.pi:
                    self.vel_cmd.linear.x = 0.0
                    self.vel_cmd.angular.z = np.minimum(abs(angle_error), 0.4)
                else:
                    self.vel_cmd.linear.x = 0.0
                    self.vel_cmd.angular.z = -np.minimum(abs(angle_error), 0.4)
            else:
                if self.yaw <= angle_to_goal + np.pi and self.yaw > angle_to_goal:
                    self.vel_cmd.linear.x = 0.0
                    self.vel_cmd.angular.z = -np.minimum(abs(angle_error), 0.4)
                else:
                    self.vel_cmd.linear.x = 0.0
                    self.vel_cmd.angular.z = np.minimum(abs(angle_error), 0.4)
            # Publish and set loop rate
            self.cmd_vel_pub.publish(self.vel_cmd)
            self.r.sleep()
            # Calculate angle error again before loop condition is checked
            angle_error = self.yaw - angle_to_goal

        # Stop rotation
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)
        if self.debug:
            print("Stopping the turn")

        ''' Then use a PID that controls the cmd velocity and drives the distance error to zero '''
        error_x = goal.x - self.x
        error_y = goal.y - self.y
        distance_error = np.sqrt(error_x**2 + error_y**2)
        angle_to_goal = np.arctan2(error_y, error_x)
        angle_error = abs(self.yaw - angle_to_goal)
        previous_distance_error = 0
        total_distance_error = 0
        previous_angle_error = 0
        total_angle_error = 0

        kp_distance = 1
        ki_distance = 0.1
        kd_distance = 0.1

        kp_angle = 1
        ki_angle = 0.1
        kd_angle = 0.1

        if self.debug:
            print("Starting the PID")

        while distance_error > 0.05:
            error_x = goal.x - self.x
            error_y = goal.y - self.y
            distance_error = np.sqrt(error_x**2 + error_y**2)
            angle_to_goal = np.arctan2(error_y, error_x)
            angle_error = abs(self.yaw - angle_to_goal)

            total_distance_error = total_distance_error + distance_error
            total_angle_error = total_angle_error + angle_error
            diff_distance_error = distance_error - previous_distance_error
            diff_angle_error = angle_error - previous_angle_error

            control_signal_distance = kp_distance*distance_error + ki_distance*total_distance_error + kd_distance*diff_distance_error
            control_signal_angle = kp_angle*angle_error + ki_angle*total_angle_error + kd_angle*diff_angle_error

            self.vel_cmd.linear.x = np.minimum(control_signal_distance, 0.1)
            self.vel_cmd.angular.z = control_signal_angle

            if angle_to_goal >= 0:
                if self.yaw <= angle_to_goal and self.yaw >= angle_to_goal - np.pi:
                    self.vel_cmd.angular.z = np.minimum(abs(control_signal_angle), 0.4)
                else:
                    self.vel_cmd.angular.z = -np.minimum(abs(control_signal_angle), 0.4)
            else:
                if self.yaw <= angle_to_goal + np.pi and self.yaw > angle_to_goal:
                    self.vel_cmd.angular.z = -np.minimum(abs(control_signal_angle), 0.4)
                else:
                    self.vel_cmd.angular.z = np.minimum(abs(control_signal_angle), 0.4)

            previous_distance_error = distance_error
            previous_angle_error = angle_error

            self.cmd_vel_pub.publish(self.vel_cmd)
            self.r.sleep()

        # Stop motion
        self.cmd_vel_pub.publish(Twist())
        if self.debug:
            print("Stopping PID")

        # Then finally rotate to the desired final yaw
        angle_error = self.yaw - final_yaw

        if self.debug:
            print("Starting to rotate towards goal orientation")

        while abs(angle_error) > 0.01:
            ''' Only useful if there is some slip/slide of the turtlebot while rotating '''
            # error_x = goal.x - self.x
            # error_y = goal.y - self.y
            # angle_to_goal = np.arctan2(error_y, error_x) # # #
            angle_error = self.yaw - final_yaw
            if self.debug:
                print("Angle to goal: {:.5f},   Yaw: {:.5f},   Angle error: {:.5f}".format(final_yaw, self.yaw, angle_error))
            if final_yaw >= 0:
                if self.yaw <= final_yaw and self.yaw >= final_yaw - np.pi:
                    self.vel_cmd.linear.x = 0.0
                    self.vel_cmd.angular.z = np.minimum(abs(angle_error), 0.4)
                else:
                    self.vel_cmd.linear.x = 0.0
                    self.vel_cmd.angular.z = -np.minimum(abs(angle_error), 0.4)
            else:
                if self.yaw <= final_yaw + np.pi and self.yaw > final_yaw:
                    self.vel_cmd.linear.x = 0.0
                    self.vel_cmd.angular.z = -np.minimum(abs(angle_error), 0.4)
                else:
                    self.vel_cmd.linear.x = 0.0
                    self.vel_cmd.angular.z = np.minimum(abs(angle_error), 0.4)
            # Publish and set loop rate
            self.cmd_vel_pub.publish(self.vel_cmd)
            self.r.sleep()
            # Calculate angle error again before loop condition is checked
            angle_error = self.yaw - final_yaw

        # Stop rotation
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)
        if self.debug:
            print("Stopping the turn")

        # Stop motion
        self.cmd_vel_pub.publish(Twist())
        if self.debug:
            print("Stopping motion")
            print("Position is currently: ({:.5f},{:.5f})    Yaw is currently: [{:.5f}]".format(self.x, self.y, self.yaw))

        print("** Waypoint Reached **")

# ------------------ End Class Definitions --------------------

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

# Adjusted 'main' to run using unicycle dynamics
if __name__ == '__main__':

    np.random.seed(0)

    rospy.init_node("robot_control_0_node", anonymous=True)
    wait_for_time()

    # Create velocity controllers
    robot_name='tb3_0'
    vel_controller_0 = VelocityController('/tb3_0/odom', '/tb3_0/cmd_vel')
    vel_controller_1 = VelocityController('/tb3_1/odom', '/tb3_1/cmd_vel')
    vel_controller_2 = VelocityController('/tb3_2/odom', '/tb3_2/cmd_vel')
    vel_controller_3 = VelocityController('/tb3_3/odom', '/tb3_3/cmd_vel')
    rospy.sleep(1.0)

    # Create the obstacle map
    # NOTE THAT:
    # obstacle 1 maps to 'red'
    # obstacle 2 maps to 'green'
    # obstacle 3 maps to 'blue'
    # FOR NOW, ASSUME THAT WE ARE ALWAYS USING 3 OBSTACLES!
    obstacle_tracker = object_map()

    ''' Set up the safety and attention environment to perform the planning in,
        here we are approximating unicycle dynamics'''

    # Set the initial point of the robotic agent in the Gazebo world (make sure this
    # is the same as the initial position in the Safety and Attention environment
    rob_heading_ang = 0
    init_point_0 = [Point(-2.5, -0.75, None), rob_heading_ang]   # the list is [Point(x,y,z),heading angle
                                                                # (measured counter-clockwise from x-axis)]
    vel_controller_0.go_to_point(init_point_0)

    # Observation strategy of the robot. Either "bin" or "sum" (sum is the preferred choice)
    rob_obs_strat = "sum"

    # Robotic agent follows linear dynamics x_k+1 = A*x_k + B*u_k
    rob_A_mat = np.eye(2)

    # Field of view of robot, assumed symmetric about current heading angle
    max_heading_view = 0.541

    # Max velocity input
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
    planning_horizon = 30

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
    sampling_time = 1.

    # Parameters weighting how to weight turning towards obstacle vs reaching goal state
    discount_factor = 0.7
    discount_weight = 15

    # Now that we have all of the ingredients, create the robot safety and
    # attention environment
    robotic_agent_environ = RobotSaAEnvironment(goal_state, goal_tolerance, beta,
                                                planning_horizon, rob_init_pos, rob_A_mat,
                                                obs_field_of_view_rad, obs_interval, rob_state_x_max,
                                                rob_state_y_max, sampling_time, rob_obs_strat,
<<<<<<< Updated upstream
                                                max_heading_view, rob_max_velocity, rob_max_turn_rate,
                                                rob_agg_turn_rate, most_rel_obs_ind, num_turning_rates,
                                                turning_rates_array, rob_heading_ang)
=======
                                                max_heading_view, rob_min_velocity, rob_max_velocity,
                                                rob_max_turn_rate, rob_agg_turn_rate, most_rel_obs_ind,
                                                num_turning_rates, turning_rates_array, rob_heading_ang,
                                                discount_factor, discount_weight)
>>>>>>> Stashed changes

    # Add the first obstacle
    # obs_1_init = np.array([-1.4, -0.3])
    obs_1_init = np.array([[-1.75], [0.25]])
    obs_1_A_matrix = np.eye(2)
    obs_1_F_matrix = np.eye(2)
    obs_1_mean_vec = np.array([[0.20],[0.0]])
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
    rdy.wait_for_ready()
    # print("Robot {} made it past Ready Check *".format(int(robot_name[-1]))) # Comment when done testing
    # sys.exit() # Comment when done testing

    # Now, while we have not reached the target point, continue executing the controller
    solve_optimization_times = []
    travel_to_point_times = []
    while robotic_agent_environ.continue_condition:

        # Resolve the problem and extract the next state to transition to, pass
        # this value to the velocity controller
        tic_sop_1 = time.time()
        robotic_agent_environ.solve_optim_prob_and_update(pool)
        solve_optimization_times.append(time.time() - tic_sop_1)

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

        next_point = Point(float(next_p[0]), float(next_p[1]), None)
        next_state = [next_point,next_yaw]

        rdy.set_ready(False)
        tic_gtp_1 = time.time()
        vel_controller_0.go_to_point(next_state)
        travel_to_point_times.append(time.time() - tic_gtp_1)
        rdy.set_ready(True)
        # Wait for the agent and the obstacles to have synchronized to their next state
        rdy.wait_for_ready()
        print("Robot {} is moving to the next waypoint *".format(int(robot_name[-1])))

        # Query the current position of each obstacle
        obs_1_x = vel_controller_1.x
        obs_1_y = vel_controller_1.y
        obs_2_x = vel_controller_2.x
        obs_2_y = vel_controller_2.y
        obs_3_x = vel_controller_3.x
        obs_3_y = vel_controller_3.y
        obs_1_cur_loc = np.array([[obs_1_x], [obs_1_y]])
        obs_2_cur_loc = np.array([[obs_2_x], [obs_2_y]])
        obs_3_cur_loc = np.array([[obs_3_x], [obs_3_y]])
        obs_act_positions = [obs_1_cur_loc, obs_2_cur_loc, obs_3_cur_loc]

        # Query camera to determine which obstacles are in view
        obs_in_view = obstacle_tracker.which_obj()

        print(obs_in_view)

        # Convert obs in view from color to number of obtacle
        obs_in_view_list = []
        if 'red' in obs_in_view:
            obs_in_view_list.append(0)
        if 'green' in obs_in_view:
            obs_in_view_list.append(1)
        if 'blue' in obs_in_view:
            obs_in_view_list.append(2)

        # Now, update the simulated and actual positions of the robot, obstacles.
        robotic_agent_environ.update_obs_positions_and_plots(obs_act_positions,obs_in_view_list)
        robotic_agent_environ.rob_pos = np.array([[vel_controller_0.x], [vel_controller_0.y]])
        robotic_agent_environ.heading_angle = vel_controller_0.yaw

        # print('----------')
        # print(robotic_agent_environ.most_rel_obs_ind)
        # print(robotic_agent_environ.heading_angle)
        # print(robotic_agent_environ.best_gamma_ind)
        # print(robotic_agent_environ.heading_angle_sequence)
        # print('----------')

    np.savetxt("optimization_times_sum.csv", np.array(solve_optimization_times),delimiter=',')
    np.savetxt("go_to_point_times.csv", np.array(travel_to_point_times),delimiter=',')