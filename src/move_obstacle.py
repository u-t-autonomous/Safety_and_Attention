#! /usr/bin/env/ python2

import rospy
import numpy as np
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, PoseStamped
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion
from Safety_and_Attention.msg import Ready
import time
import command_control as cc

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

        # Then finally rotate to the desired final yaw if final_yaw is not None
        if final_yaw:
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
    rospy.init_node("object_control_node", anonymous=True)
    wait_for_time()

    robot_name = 'tb3_1'
    # Create velocity controllers
    vel_controller = VelocityController('/tb3_1/odom', '/tb3_1/cmd_vel', debug=True)
    rospy.sleep(1.0)

    obs_traj = np.load('obstacle_{}_trajectory.npy'.format(int(robot_name[-1])))
    rdy = cc.ReadyTool(robot_name)


    for i in range(np.shape(obs_traj)[0]):
        print('x is: {}, y is: {}'.format(obs_traj[i,0], obs_traj[i,1]))
        point_0 = [Point(obs_traj[i][0], obs_traj[i][1], None),None] # r_mp

        if robot_name == 'tb3_1':
            wp = [Point(point_0[0].x - -1.75, point_0[0].y - 0.25, None), None]
        elif robot_name == 'tb3_2':
            wp = [Point(point_0[0].x - 1.20, point_0[0].y - 1.80, None), None]
        elif robot_name == 'tb3_3':
            wp = [Point(point_0[0].x - 2.25, point_0[0].y - -2.0, None), None]
        else:
            rospy.logerr("ERROR - Transformation to the odemetry frame could not be completed. The value of the robot name is {}".format(robot_name))
            sys.exit()

        rdy.set_ready(True)
        print("*** Robot {} is ready and waiting to start ***".format(int(robot_name[-1])))
        rdy.wait_for_ready()
        rdy.set_ready(False)
        vel_controller.go_to_point(wp)
