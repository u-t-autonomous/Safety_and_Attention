#! /usr/bin/env/ python2
import rospy
import numpy as np
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, PoseStamped, Quaternion, Vector3
from tf.transformations import euler_from_quaternion
import command_control as cc
# import simplejson as json
# from reactive_control_utils import parseJson, next_game_and_agent_state, powerset
# from grid_state_converter import make_grid_converter
import ViconTracker
# import copy
# from std_msgs.msg import ColorRGBA
# from visualization_msgs.msg import Marker


def correctAngle(angle):
    if angle:
        if angle > np.pi:
            return correctAngle(angle-2*np.pi)
        elif angle < -np.pi:
            return correctAngle(angle+2*np.pi)
    return angle

def min_mag(PID_signal,ceiling_mag):
    if np.fabs(PID_signal) < np.fabs(ceiling_mag):
        return PID_signal
    return np.sign(PID_signal)*ceiling_mag

# ------------------ Start Class Definitions ------------------

class TBVelocityController:
    """Simple velocity controller meant to be used with turtlebot3"""
    # Do not set track_name var if using odom
    def __init__(self, odom_topic_name, cmd_vel_topic_name, tracker = None, track_name = None, debug=False):
        self.tracker = tracker
        if track_name:
            self.track_ind = int(track_name[-1]) # 'xxx_0' is required format
        self.debug = debug
        self.__odom_sub = rospy.Subscriber(odom_topic_name, Odometry, self.__odomCB)
        self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic_name, Twist, queue_size = 1)

        self.x = 0
        self.y = 0
        self.yaw = 0
        self.r = rospy.Rate(4)
        self.vel_cmd = Twist()

    def __odomCB(self, msg):
        if not self.tracker:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            rot_q = msg.pose.pose.orientation
            _, _, self.yaw = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

        else:
            self.x = self.tracker.data[self.track_ind].translation.x
            self.y = self.tracker.data[self.track_ind].translation.y
            rot_q  = self.tracker.data[self.track_ind].rotation
            _, _, self.yaw = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

    def go_to_point(self, data):
    	# input variable goal should be of type [geometry_msgs/Point,yaw]
        goal = data#[0]

        self.vel_cmd.linear.x = 0
        self.vel_cmd.linear.y = 0
        self.vel_cmd.linear.z = 0

        self.vel_cmd.angular.x = 0
        self.vel_cmd.angular.y = 0
        self.vel_cmd.angular.z = 0

        self.cmd_vel_pub.publish(self.vel_cmd)

        # final_yaw = data[1]

        print("Starting to head towards the waypoint")

        # ''' First do the rotation towards the goal '''
        # error_x = goal.x - self.x
        # error_y = goal.y - self.y
        # angle_to_goal = np.arctan2(error_y, error_x)
        # angle_error = self.yaw - angle_to_goal

        if self.debug:
            print("Starting to rotate towards waypoint")
        dist_tol = .05
        ang_tol = np.pi/180
        max_vel = .12
        max_ang_vel = np.pi/10

        kp_lin = 1
        ki_lin = 0
        kd_lin = .5

        kp_ang = 1.5
        ki_ang = 0
        kd_ang = 1.75

        distance_error = np.sqrt((goal.x - self.x)**2 + (goal.y - self.y)**2)
        angle_error = correctAngle(np.arctan2(goal.y - self.y, goal.x - self.x) - correctAngle(self.yaw))
        previous_distance_error = distance_error
        previous_angle_error = angle_error
        total_distance_error = 0
        total_angle_error = 0
        # while abs(angle_error) > ang_tol:
        #     angle_error = (correctAngle(np.arctan2(goal.y - self.y, goal.x - self.x) - correctAngle(self.yaw)) + previous_angle_error)/2
        #     total_angle_error+=(angle_error+previous_angle_error)/2
        #     self.vel_cmd.angular.z = min_mag((kp_ang*angle_error+ki_ang*total_angle_error+kd_ang*(angle_error - previous_angle_error)), max_ang_vel)
        #     previous_angle_error = angle_error
        #     self.cmd_vel_pub.publish(self.vel_cmd)
        #     self.r.sleep()

        while distance_error > dist_tol:
            # if self.debug:
            # print("Angle to goal: {:.5f},   Yaw: {:.5f},   Angle error: {:.5f}".format(angle_to_goal, self.yaw, angle_error))
            print("error x: {:.5f}, error y: {:.5f} error theta: {:.5f}".format(goal.x - self.x,goal.y - self.y, angle_error))
            distance_error = np.sqrt((goal.x - self.x)**2 + (goal.y - self.y)**2)
            angle_error = correctAngle(np.arctan2(goal.y - self.y, goal.x - self.x) - correctAngle(self.yaw))# + previous_angle_error)/2
            total_distance_error+=(distance_error+previous_distance_error)/2
            total_angle_error+=(angle_error+previous_angle_error)/2

            self.vel_cmd.angular.z = min_mag((kp_ang*angle_error+ki_ang*total_angle_error+kd_ang*(angle_error - previous_angle_error)), max_ang_vel)
            self.vel_cmd.linear.x = np.minimum((kp_lin*distance_error+ki_lin*total_distance_error+kd_lin*(distance_error - previous_distance_error)), max_vel*(np.fabs(np.pi-angle_error)/np.pi)**.5)

            print("ang cmd: {:.5f}, lin cmd: {:.5f}".format(min_mag((kp_ang*angle_error+ki_ang*total_angle_error+kd_ang*(angle_error - previous_angle_error)), max_ang_vel),np.minimum((kp_lin*distance_error+ki_lin*total_distance_error+kd_lin*(distance_error - previous_distance_error)), max_vel*(np.fabs(np.pi-angle_error)/np.pi)**.5)))

            previous_distance_error = distance_error
            previous_angle_error = angle_error

            self.cmd_vel_pub.publish(self.vel_cmd)
            self.r.sleep()

        self.cmd_vel_pub.publish(Twist())
        if self.debug:
            print("Stopping motion")
            print("Position is currently: ({:.5f},{:.5f})    Yaw is currently: [{:.5f}]".format(self.x, self.y, self.yaw))

        print("** Waypoint Reached **")

# ------------------ End Class Definitions --------------------

