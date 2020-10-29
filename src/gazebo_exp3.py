#! /usr/bin/env/ python2

"""
    Description:    Velocity controller node for experiment with Mahsa
    Author:         Jesse Quattrociocchi
    Created:        May-July 2019
"""

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
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import LaserScan, PointCloud2
from tf.transformations import euler_from_quaternion
# from grid_state_converter import *
# Imports for Algorithm side
import copy
import random
# from partial_semantics import *
# Rviz
# from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Quaternion, Vector3


# To import for running safety and attention simulations
from linear_dynamics_auxilliary_functions import (ellipsoid_level_function,
    check_collisions,continue_condition,get_mu_sig_over_time_hor,
    get_mu_sig_theta_over_time_hor,get_Q_mats_over_time_hor,
    get_Qplus_mats_over_time_hor,propagate_linear_obstacles,
    propagate_nonlinear_obstacles)
from dc_based_motion_planner import (create_dc_motion_planner,
    solve_dc_motion_planning,create_cvxpy_motion_planner_obstacle_free)
from dc_motion_planning_ecos import (dc_motion_planning_call_ecos,
    stack_params_for_ecos,set_up_independent_ecos_params)
from Robot_SaA_Environment import LinearObstacle, NonlinearObstacle, RobotSaAEnvironment
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Ellipse

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

    def go_to_point(self, goal):

        # input variable goal should be of type geometry_msgs/Point

        print("Starting to head towards the waypoint")

        ''' First do the rotation towards the goal '''
        error_x = goal.x - self.x
        error_y = goal.y - self.y
        angle_to_goal = np.arctan2(error_y, error_x)
        angle_error = self.yaw - angle_to_goal

        if self.debug:
            print("Starting to rotate towards waypoint")

        while abs(angle_error) > 0.05:
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
            print("Position is currently: ({:.5f},{:.5f})    Yaw is currently: [{:.5f}]".format(self.x, self.y, self.yaw))

        print("** Waypoint Reached **")

class Scan(object):
    ''' Scan object that holds the laser scan and point cloud of the input topic '''
    def __init__(self, scan_topic_name):
        self.raw = LaserScan()
        self.cloud = None
        self.lp = lg.LaserProjection()
        self.__scan_sub = rospy.Subscriber(scan_topic_name, LaserScan, self.__scanCB)

    def __scanCB(self, msg):
        self.raw = msg
        self.cloud = self.lp.projectLaser(msg)

    def print_scan(self, scan_type='cloud'):
        if not isinstance(scan_type, str):
            print("The scan_type variable must be a string (raw, cloud)")
        if scan_type == 'cloud':
            print("The point cloud is: \n{}".format(self.cloud))
        elif scan_type == 'raw':
            print("the raw message is: \n{}".format(self.raw))
        else:
            print("The scan types are: raw, cloud")

    def pc_generator(self, field_names=('x', 'y', 'z')):
        return pc2.read_points(self.cloud, field_names)


class Scanner(Scan):
    ''' Converts a point cloud into a set of states that describe the state of a gridworld '''
    def __init__(self, scan_topic_name, grid_converter, debug=False):
        super(Scanner, self).__init__(scan_topic_name)
        self.grid_converter = grid_converter
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.debug = debug


    def convert_pointCloud_to_gridCloud(self, pc):
        ''' The input is a pointcloud2 generator, where each item is a tuple '''
        if not isinstance(pc, types.GeneratorType):
            print("The input must be a pointcloud2 generator (not the generator function)")
            sys.exit()
        states = set()
        for item in pc:
            if self.debug: print(item)
            new_pose = self.transform_coordinates(item)
            if self.debug: print(new_pose)
            if not (self.grid_converter.base.x < new_pose.pose.position.x < self.grid_converter.maximum.x) or not (self.grid_converter.base.y < new_pose.pose.position.y < self.grid_converter.maximum.y):
                continue
            else:
                pcPoint = Point()
                pcPoint.x = new_pose.pose.position.x
                pcPoint.y = new_pose.pose.position.y
                pcPoint.z = new_pose.pose.position.z
                states.add(self.grid_converter.cart2state(pcPoint))

        return states

    def transform_coordinates(self, coord, from_frame='base_scan', to_frame='odom'):
        ''' Gets frame transform at latest time '''
        p1 = PoseStamped()
        p1.header.frame_id = from_frame
        p1.pose.position.x = coord[0]
        p1.pose.position.y = coord[1] # Not sure this is right
        p1.pose.position.z = coord[2]
        p1.pose.orientation.w = 1.0    # Neutral orientation
        transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time(0), rospy.Duration(1.0))
        return tf2_geometry_msgs.do_transform_pose(p1, transform)


class ColorMixer(object):
    ''' Finds the color mix between two colors.
        color_0 and color_1 are chosen from:
            {'red', 'green', 'blue'}. '''
    def __init__(self, color_0, color_1):
        colors = ('red', 'green', 'blue')
        if not color_0 in colors or not color_1 in colors:
            rospy.logerr("Choose a color from ('red', 'green', 'blue')")
            sys.exit()

        self._c0 = self.__determine_input(color_0)
        self._c1 = self.__determine_input(color_1)
        self.last_c = []

    ''' Input is a double on [0-1] where:
        0 maps to c_0 = 255 and c_1 = 0,
        1 maps to c_0 = 0 and c_1 = 255. '''
    def __determine_input(self, color):
        if color == 'red':
            return [255, 0, 0]
        elif color == 'green':
            return [0, 255, 0]
        elif color == 'blue':
            return [0, 0, 255]
        else:
            rospy.logerr("Choose a color from ('red', 'green', 'blue')")
            sys.exit()

    def __check_mixing_value(self, val):
        if not (0 <= val <= 1):
            rospy.logerr("get_color value must be between [0-1]")
            sys.exit()

    def get_color(self, val):
        self.__check_mixing_value(val)
        self.last_c = [val*(self._c1[j] - self._c0[j]) + self._c0[j] for j in range(3)]
        return self.last_c

    def get_color_norm(self, val):
        self.__check_mixing_value(val)
        self.last_c = [val*(self._c1[j] - self._c0[j]) + self._c0[j] for j in range(3)]
        self.last_c[:] = [x/255 for x in self.last_c]
        return self.last_c


class BeliefMarker(object):
    def __init__(self):
        self.pub = rospy.Publisher('visualization_marker', Marker, queue_size=5)
        rospy.sleep(0.5)
        self.marker = Marker()
        self.marker.header.frame_id = 'odom' # Change to odom for experiment
        self.marker.header.stamp = rospy.get_rostime()
        self.marker.id = 0
        self.marker.type = Marker.CUBE_LIST
        # self.marker.action = Marker.ADD   # Not sure what this is for???
        self.marker.points = []
        self.marker.pose.orientation = Quaternion(0,0,0,1)
        self.marker.scale = Vector3(1,1,0.05)
        self.marker.colors = []
        self.marker.lifetime = rospy.Duration(0)

    def show_marker(self):
        self.pub.publish(self.marker)
        rospy.sleep(0.5)


# ------------------ End Class Definitions --------------------

# ------------------ Start Function Definitions ---------------

def wait_for_time():
    """Wait for simulated time to begin """
    while rospy.Time().now().to_sec() == 0:
        pass


def get_direction_from_key_stroke():
    while(True):
        val = raw_input("Enter command {a = left, w = up , s = down, d = right, h = hold, k = exit}: ")
        if val == 'w':
            print("You pressed up")
            return 'up'
        elif val == 's':
            print("You pressed down")
            return 'down'
        elif val == 'a':
            print("You pressed left")
            return 'left'
        elif val == 'd':
            print("You pressed right")
            return 'right'
        elif val == 'h':
            print("You pressed hold")
            return 'hold'
        elif val == 'k':
            print("You chose to exit")
            print("Closing program")
            sys.exit()
        else:
            print("You must enter a valid command {a = left, w = up , s = down, d = right, h = hold, k = exit}")


def next_waypoint_from_direction(direction, current_pose):
    """ Changes in x are from Left/Right action.
        CHanges in y are from Up/Down action.
        Set the next wp to the center of 1x1 cells (i.e x=1.5, y=1.5).
        Error logging uses ROS.
        direction is a string. current_pose is a Point object """

    wp = Point(current_pose.x, current_pose.y, None)
    if direction == 'up':
        wp.y = np.ceil(wp.y) + 0.5
    elif direction == 'down':
        wp.y = np.floor(wp.y) - 0.5
    elif direction == 'left':
        wp.x = np.floor(wp.x) - 0.5
    elif direction == 'right':
        wp.x = np.ceil(wp.x) + 0.5
    else:
        err_msg = "The direction value {} is not valid".format(direction)
        rospy.logerr(err_msg)
        sys.exit()

    return wp


def move_TB_keyboard(controller_object):
    """ Function to wrap up the process of moving the turtlebot via key strokes.
        Requires ROS to be running and a controller object """
    pose = Point(controller_object.x, controller_object.y, None)
    dir_val = get_direction_from_key_stroke()
    if dir_val == 'hold':
        print("* You chose to hold *")
    else:
        wp = next_waypoint_from_direction(dir_val, pose)
        controller_object.go_to_point(wp)


def move_TB(controller_object, dir_val):
    """ Function to wrap up the process of moving the turtlebot via input.
        Requires ROS to be running and a controller object """
    pose = Point(controller_object.x, controller_object.y, None)
    if dir_val == 'hold':
        print("* You chose to hold *")
    else:
        wp = next_waypoint_from_direction(dir_val, pose)
        controller_object.go_to_point(wp)


def make_grid_converter(grid_vars):
    ''' Returns a Grid class object '''
    base = Point()
    base.x = grid_vars[0]
    base.y = grid_vars[1]
    base.z = 0 # Used for the RVIZ marker
    maximum = Point()
    maximum.x = grid_vars[2]
    maximum.y = grid_vars[3]
    converter = Grid(grid_vars[5],grid_vars[4],base,maximum)
    return converter


def show_converter(controller):
    cart = Point()
    cart.x = controller.x
    cart.y = controller.y
    cart.z = 0
    print("You are now in State: {}".format(grid_converter.cart2state(cart)))


def get_visible_points_circle(center, radius):
    theta = np.linspace(0, 2*np.pi, 360)
    rays = np.linspace(0, radius, 35)
    vis_points = set()
    for angle in theta:
        for r in rays:
            x = center[0] + r*np.cos(angle)
            y = center[1] + r*np.sin(angle)
            vis_points.add((x,y))
    return vis_points


def get_vis_states_set(current_loc, converter, vis_dis=3.5):
    vis_points = get_visible_points_circle(current_loc, vis_dis)
    s = set()
    a_point = Point()
    for item in vis_points:
        if not (converter.base.x < item[0] < converter.maximum.x) or not (converter.base.y < item[1] < converter.maximum.y):
            continue
        else:
            a_point.x = item[0]
            a_point.y = item[1]
            a_point.z = 0
            s.add(converter.cart2state(a_point))

    return s


def get_occluded_states_set(controller, scanner, vis_dis=3.5):
    occl = set()
    temp = set()
    # Get all the points along a ray passing through each point in the pointcloud
    for item in scanner.pc_generator(field_names=('x', 'y', 'z','index')):
        angle = item[3]*scanner.raw.angle_increment
        radius = item[0]/np.cos(angle) # x/cos(theta)
        rays = np.linspace(radius, vis_dis, (vis_dis-radius)/0.1)
        for r in rays:
            x = r*np.cos(angle)
            y = r*np.sin(angle)
            temp.add((x,y,0)) # z is here for the transformation later on

    # Convert those points included in the set to the odom frame and make a new set
    for item in temp:
        new_pose = scanner.transform_coordinates(item)
        if not (scanner.grid_converter.base.x < new_pose.pose.position.x < scanner.grid_converter.maximum.x) or not (scanner.grid_converter.base.y < new_pose.pose.position.y < scanner.grid_converter.maximum.y):
            continue
        else:
            occlPoint = Point()
            occlPoint.x = new_pose.pose.position.x
            occlPoint.y = new_pose.pose.position.y
            occlPoint.z = new_pose.pose.position.z
            occl.add(scanner.grid_converter.cart2state(occlPoint))

    return occl


def make_array(scan, vis, occl, array_shape):
    ''' Assumes array_shape is (row,col).
        This does not check for overlapping states in the vis set and the scan set.
        It should work though bc of order. '''
    a = -np.ones(array_shape)
    real_occl = occl - occl.intersection(scan)
    real_vis = vis - vis.intersection(real_occl)

    for v in real_vis:
        row_ind = v / array_shape[1]
        col_ind = v % array_shape[1]
        a[row_ind,col_ind] = 0

    for s in scan:
        row_ind = s / array_shape[1]
        col_ind = s % array_shape[1]
        a[row_ind,col_ind] = 1

    return a


def make_user_wait(msg="Enter exit to exit"):
    data = raw_input(msg + "\n")
    if data == 'exit':
        sys.exit()

# ------------------ End Function Definitions -----------------

if __name__ == '__main__':

    np.random.seed(0)

    rospy.init_node("velocity_controller")
    wait_for_time()

    # make_user_wait('Press enter to start')

    # Create velocity controller and converter objects
    vel_controller_0 = VelocityController('/tb3_0/odom', '/tb3_0/cmd_vel')
    vel_controller_1 = VelocityController('/tb3_1/odom', '/tb3_1/cmd_vel')
    vel_controller_2 = VelocityController('/tb3_2/odom', 'tb3_2/cmd_vel')
    rospy.sleep(1.0)

    # Set the initial point of the robotic agent in the Gazebo world (make sure this
    # is the same as the initial position in the Safety and Attention environment
    init_point_0 = Point(-8, -8, None)
    init_point_1 = Point(-7.65, -5.2, None)
    init_point_2 = Point(3.4,2.8, None)
    vel_controller_0.go_to_point(init_point_0)
    vel_controller_1.go_to_point(init_point_1)
    vel_controller_2.go_to_point(init_point_2)

    # make_user_wait()

    # Set up the safety and attention environment to perform the planning in

    # Observation strategy of the robot. Either "bin" or "sum" (sum is preferred choice)
    rob_obs_strat = "sum"

    # Robotic agent follows linear dynamics x_k+1 = A*x_k + B*u_k
    rob_A_mat = np.eye(2)
    rob_B_mat = np.eye(2)

    # Define the state that the robotic agent must travel to and the L2-norm
    # tolerance for which we can say that the agent "reached" that state
    goal_state = np.array([6,6])
    goal_tolerance = 5e-2

    # Parameter for constructing Gaussian level sets of obstacle positions
    # A lower beta means that the "keep-out region" is larger
    beta = 0.01

    # Number of time steps into the future that the robotic agent must plan
    planning_horizon = 25

    # Initial position of the robotic agent in the environment
    rob_init_pos = np.array([-8, -8])

    # The size of the circle (assumed to be in meters?) for which the robotic
    # agent can make an observation about an obstacle if the obstacle is within
    # that position.
    obs_field_of_view_rad = 100

    # The number of time steps between subsequent opportunities for the robotic
    # agent to make an observation
    obs_interval = 1

    # Assuming a square, the absolute value in the L1 sense that the position of
    # the robotic agent can be in any of the cardinal directions
    rob_state_max = 10

    # This parameter is for how large of "steps" the difference in waypoints can be,
    # not really a physical parameter and we will likely need to adjust this
    rob_input_max = 10

    # Time between subsequent time steps "k" and "k+1" (again, not really a physical
    # parameter, will probably need to play around with this value)
    sampling_time = 0.05

    # Now that we have all of the ingredients, create the robot safety and
    # attention environment
    robotic_agent_environ = RobotSaAEnvironment(goal_state, goal_tolerance, beta,
                planning_horizon, rob_init_pos, rob_A_mat, rob_B_mat, obs_field_of_view_rad,
                obs_interval, rob_state_max, rob_input_max, sampling_time, rob_obs_strat)

    # Add the first obstacle
    obs_1_init = np.array([-7.65,-5.2])
    obs_1_A_matrix = np.eye(2)
    obs_1_F_matrix = np.eye(2)
    obs_1_mean_vec = np.array([7.5,7.4])
    obs_1_cov_mat = np.array([[2,0.25],[0.25,2]])
    obs_1_radius = 1.2
    robotic_agent_environ.add_linear_obstacle(obs_1_init,obs_1_A_matrix,
        obs_1_F_matrix,obs_1_mean_vec,obs_1_cov_mat,obs_1_radius)

    # Add the second obstacle
    obs_2_init = np.array([3.4, 2.8])
    obs_2_A_matrix = np.eye(2)
    obs_2_F_matrix = np.eye(2)
    obs_2_mean_vec = np.array([-8.8, -8.8])
    obs_2_cov_mat = np.array([[70, 18], [18, 70]])
    obs_2_radius = 0.5
    robotic_agent_environ.add_linear_obstacle(obs_2_init, obs_2_A_matrix,
        obs_2_F_matrix, obs_2_mean_vec, obs_2_cov_mat, obs_2_radius)

    # Now, while we have not reached the target point, continue executing the controller
    while robotic_agent_environ.continue_condition:

        # Resolve the problem
        robotic_agent_environ.solve_optim_prob_and_update()

        # Agent
        next_p = robotic_agent_environ.nominal_trajectory[:,0]
        next_point = Point(float(next_p[0]),float(next_p[1]),None)

        # Obstacle 1
        obs_1_n = robotic_agent_environ.lin_obs_list[0].act_position
        obs_1_np = Point(float(obs_1_n[0]),float(obs_1_n[1]),None)

        # Obstacle 2
        obs_2_n = robotic_agent_environ.lin_obs_list[1].act_position
        obs_2_np = Point(float(obs_2_n[0]),float(obs_2_n[1]),None)

        # Update controllers
        vel_controller_0.go_to_point(next_point)
        vel_controller_1.go_to_point(obs_1_np)
        vel_controller_2.go_to_point(obs_2_np)

        # Update the robotic agent's position
        robotic_agent_environ.rob_pos = np.array([vel_controller_0.x, vel_controller_0.y])
