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
# from partial_semantics import *
# Rviz
# from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Quaternion, Vector3

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

# FOR MAPPING USING THE VICON AND TURTLEBOT CAMERA SYSTEM
from detect_colors_hardware import object_map

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

class ReadyTool:
    """Tool to help control the executions of multiple nodes from a master node"""
    def __init__(self, robot_name='tb3_0', ready_topic_name='/ready_start_cmd'):
        # self.ready = False
        # self.__ready_wait_sub = rospy.Subscriber(ready_topic_name, Bool, self.__readyCB )
        ''' **** PUT A PUBLISHER HERE **** '''
        # self.__pub = rospy.Publisher('/ready_start_cmd', Bool, queue_size=1)


        # Set up flags for sim start as well as Set Ready start value
        self.flag_val = False
        self.ready2start = Ready()
        self.ready2start.name = robot_name
        self.ready2start.ready = False
        self.flag_pub = rospy.Publisher('/tb3_' + str(robot_name[-1]) + '/ready_start', Ready, queue_size=1)
        rospy.Subscriber('/ready_start_cmd', Bool, self.flagCB)

    def flagCB(self, msg):
        self.flag_val = msg.data

    def set_ready(self, val):
        self.ready2start.ready = val
        # self.flag_vals[self.platform_id] = val
        t_end = time.time() + 0.1
        while time.time() < t_end:
            self.flag_pub.publish(self.ready2start)

    def wait_to_move(self):
        rospy.sleep(10)

    def wait_for_ready(self):
        # check = False
        while not self.flag_val:
            # if not check:
            #     print("Waiting to start")
            #     check = True
            rospy.sleep(0.01)
        print("*** robot {} is starting ***".format(int(self.ready2start.name[-1])))
        # self.flag_val = False

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

    # For parallel processing
    freeze_support()
    pool = multiprocessing.Pool()
    #pool = ThreadPool()

    np.random.seed(0)

    rospy.init_node("robot_control_0_node", anonymous=True)
    wait_for_time()

    vicon_track = Tracker()

    # Create velocity controllers
    robot_name='tb3_0'
    vel_controller_0 = VelocityController('/tb3_0/odom', '/tb3_0/cmd_vel')
    # vel_controller_1 = VelocityController('/tb3_1/odom', '/tb3_1/cmd_vel')
    # vel_controller_2 = VelocityController('/tb3_2/odom', '/tb3_2/cmd_vel')
    # vel_controller_3 = VelocityController('/tb3_3/odom', '/tb3_3/cmd_vel')
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
    # init_point_0 = [Point(-2.5, -0.75, None), rob_heading_ang]   # the list is [Point(x,y,z),heading angle
                                                                # (measured counter-clockwise from x-axis)]
    # vel_controller_0.go_to_point(init_point_0)

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
    goal_state = np.array([1.9, 1.9])
    goal_tolerance = 10e-2

    # Parameter for constructing Gaussian level sets of obstacle positions
    # A lower beta means that the "keep-out region" is larger
    beta = 0.01

    # Number of time steps into the future that the robotic agent must plan
    planning_horizon = 12

    # Initial position of the robotic agent in the environment
    rob_init_pos = np.array([-2.5, -0.75])

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

    # Now that we have all of the ingredients, create the robot safety and
    # attention environment
    robotic_agent_environ = RobotSaAEnvironment(goal_state, goal_tolerance, beta,
                                                planning_horizon, rob_init_pos, rob_A_mat,
                                                obs_field_of_view_rad, obs_interval, rob_state_x_max,
                                                rob_state_y_max, sampling_time, rob_obs_strat,
                                                max_heading_view, rob_max_velocity, rob_max_turn_rate,
                                                rob_agg_turn_rate, most_rel_obs_ind, num_turning_rates,
                                                turning_rates_array, rob_heading_ang)

    # Add the first obstacle
    # obs_1_init = np.array([-1.4, -0.3])
    obs_1_init = np.array([-1.75, 0.25])
    obs_1_A_matrix = np.eye(2)
    obs_1_F_matrix = np.eye(2)
    obs_1_mean_vec = np.array([0.20, 0.0])
    obs_1_cov_mat = np.array([[0.008, 0.001], [0.001, 0.008]])
    obs_1_radius = 0.25
    robotic_agent_environ.add_linear_obstacle(obs_1_init, obs_1_A_matrix,
                                              obs_1_F_matrix, obs_1_mean_vec, obs_1_cov_mat, obs_1_radius)

    # Add the second obstacle
    # obs_2_init = np.array([1.3, 1.])
    obs_2_init = np.array([1.2, 1.8])
    obs_2_A_matrix = np.eye(2)
    obs_2_F_matrix = np.eye(2)
    obs_2_mean_vec = np.array([0.025, -0.1])
    obs_2_cov_mat = np.array([[0.004, 0.0015], [0.0015, 0.005]])
    obs_2_radius = 0.25
    robotic_agent_environ.add_linear_obstacle(obs_2_init, obs_2_A_matrix,
                                              obs_2_F_matrix, obs_2_mean_vec, obs_2_cov_mat, obs_2_radius)

    # Add the third obstacle
    # obs_3_init = np.array([-0.3, -1.2])
    obs_3_init = np.array([2.25, -2.0])
    obs_3_A_matrix = np.eye(2)
    obs_3_F_matrix = np.eye(2)
    obs_3_mean_vec = np.array([0.025, 0.15])
    obs_3_cov_mat = np.array([[0.005, 0.0015], [0.0015, 0.008]])
    obs_3_radius = 0.25
    robotic_agent_environ.add_linear_obstacle(obs_3_init, obs_3_A_matrix,
                                              obs_3_F_matrix, obs_3_mean_vec, obs_3_cov_mat, obs_3_radius)

    # Wait until all other robots are ready
    rdy = ReadyTool(robot_name)
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
        next_yaw = robotic_agent_environ.heading_angle_sequence[1]

        # Need to convert yaw in [0,2pi] to [-pi,pi]
        if 0<=next_yaw <= np.pi:
            next_yaw = next_yaw
        elif np.pi < next_yaw <= 2*np.pi:
            next_yaw = next_yaw - 2*np.pi
        elif 0 > next_yaw >= -np.pi:
            next_yaw = next_yaw
        else:  # Must be that the yaw angle is between -pi and -2*pi -> want between 0 and pi
            next_yaw = next_yaw + 2*np.pi

        next_point = Point(float(next_p[0])- -2.50, float(next_p[1]) - -0.75, None)
        #### MUST ADD THE TRANSFORM !!!!!!   ##### <_____________-----------------------------------------
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
        obs_1_x = vicon_track.data[1].translation.x
        obs_1_y = vicon_track.data[1].translation.y
        obs_2_x = vicon_track.data[2].translation.x
        obs_2_y = vicon_track.data[2].translation.y
        obs_3_x = vicon_track.data[3].translation.x
        obs_3_y = vicon_track.data[3].translation.y
        # obs_1_x = vel_controller_1.x
        # obs_1_y = vel_controller_1.y
        # obs_2_x = vel_controller_2.x
        # obs_2_y = vel_controller_2.y
        # obs_3_x = vel_controller_3.x
        # obs_3_y = vel_controller_3.y
        obs_1_cur_loc = np.array([obs_1_x, obs_1_y])
        obs_2_cur_loc = np.array([obs_2_x, obs_2_y])
        obs_3_cur_loc = np.array([obs_3_x, obs_3_y])
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
        robotic_agent_environ.rob_pos = np.array([vicon_track.data[0].translation.x, vicon_track.data[0].translation.y])
        rot_q = vicon_track.data[0].rotation
        _, _, vicon_yaw = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        robotic_agent_environ.heading_angle = vicon_yaw
        # robotic_agent_environ.rob_pos = np.array([vel_controller_0.x, vel_controller_0.y])
        # robotic_agent_environ.heading_angle = vel_controller_0.yaw

        print('----------')
        print(robotic_agent_environ.most_rel_obs_ind)
        print(robotic_agent_environ.heading_angle)
        print(robotic_agent_environ.best_gamma_ind)
        print(robotic_agent_environ.heading_angle_sequence)
        print('----------')

    np.savetxt("optimization_times_sum.csv", np.array(solve_optimization_times),delimiter=',')
    np.savetxt("go_to_point_times.csv", np.array(travel_to_point_times),delimiter=',')