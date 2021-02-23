#! /usr/bin/env/ python2

# Imports for ROS side
import rospy
import numpy as np
import sys
# import types
# import tf2_ros
# import tf2_geometry_msgs
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, PoseStamped
# from std_msgs.msg import Bool
# import laser_geometry.laser_geometry as lg
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import LaserScan, PointCloud2
from tf.transformations import euler_from_quaternion
from Safety_and_Attention.msg import Ready
# import time
# Imports for Algorithm side
# import copy
# import random
# from partial_semantics import *
# Rviz
# from visualization_msgs.msg import Marker
# from std_msgs.msg import ColorRGBA
# from geometry_msgs.msg import Quaternion, Vector3, TransformStamped

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

        # Then finally rotate to the desired final yaw
        angle_error = self.yaw - final_yaw

        if self.debug:
            print("Starting to rotate towards goal orientation")

        while abs(angle_error) > 0.05:
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


class ReadyTool:
    """Tool to help control the executions of multiple nodes from a master node"""
    def __init__(self, robot_name='tb3_0', ready_topic_name='/ready_start_cmd'):
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
        while not self.flag_val:
            rospy.sleep(0.01)
        print("*** robot {} is starting ***".format(int(self.ready2start.name[-1])))

# ------------------ End Class Definitions --------------------

def make_user_wait(msg="Enter exit to exit"):
    data = raw_input(msg + "\n")
    if data == 'exit':
        sys.exit()

def wait_for_time():
    """Wait for simulated time to begin """
    while rospy.Time().now().to_sec() == 0:
        pass


if __name__ == '__main__':
    rospy.init_node("robot_control_1_node", anonymous=True)
    wait_for_time()

    robot_name = 'tb3_1'

    # Create velocity controllers
    vel_controller = VelocityController('/tb3_1/odom', '/tb3_1/cmd_vel', debug=True)
    rospy.sleep(1.0)

    point_0 = [Point(1, 0, None),np.pi]

    rdy = ReadyTool(robot_name)
    print("*** Robot {} is ready and waiting to start ***".format(int(robot_name[-1])))
    rdy.set_ready(True)
    rdy.wait_for_ready()
    print("Robot {} made it past Ready Check *".format(int(robot_name[-1]))) # Comment when done testing

    vel_controller.go_to_point(point_0)

    make_user_wait()
