#! /usr/bin/env/ python2

''' Module for setting up command and control using ROS topics '''

from Safety_and_Attention.msg import Ready
from std_msgs.msg import Bool
import rospy
import time

class Commander:
    def __init__(self,  num_agents):
        # Publisher to send commands.
        self.pub = rospy.Publisher('/ready_start_cmd', Bool, queue_size=1)

        # Set up flags
        self.flag_subs = {}
        self.flag_vals = {}
        for a in range(num_agents):
            self.flag_vals[a] = False
        for a in range(num_agents):
            self.flag_subs[a] = rospy.Subscriber('/tb3_' + str(a) +  '/ready_start', Ready, self.flagCB)

    def flagCB(self, msg):
        id_num = int(msg.name[-1])
        self.flag_vals[id_num] = msg.ready

    def set_ready(self, value):
        t_end = time.time() + 2 # Publish for 2 seconds
        while time.time() < t_end:
            self.pub.publish(value)

class ReadyTool:
    """Tool to help control the executions of multiple nodes from a master node"""
    def __init__(self, robot_name='tb3_0', set_ready_delay=0.25, ready_topic_name='/ready_start_cmd'):
        # Set up flags for sim start as well as Set Ready start value
        self.flag_val = False
        self.srd = set_ready_delay
        self.ready2start = Ready()
        self.ready2start.name = robot_name
        self.ready2start.ready = False
        self.flag_pub = rospy.Publisher('/tb3_' + str(robot_name[-1]) + '/ready_start', Ready, queue_size=1)
        rospy.Subscriber(ready_topic_name, Bool, self.flagCB)

    def flagCB(self, msg):
        self.flag_val = msg.data

    def set_ready(self, val):
        self.ready2start.ready = val
        # self.flag_vals[self.platform_id] = val
        t_end = time.time() + self.srd
        while time.time() < t_end:
            self.flag_pub.publish(self.ready2start)

    def wait_to_move(self):
        rospy.sleep(10)

    def wait_for_ready(self):
        while not self.flag_val:
            rospy.sleep(0.01)
        print("*** robot {} is starting ***".format(int(self.ready2start.name[-1])))