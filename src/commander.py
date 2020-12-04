#! /usr/bin/env/ python2

from Safety_and_Attention.msg import Ready
from std_msgs.msg import Bool

import sys
import rospy
import numpy as np
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


if __name__ == "__main__":
    rospy.init_node('robot_command', anonymous=True)
    cmd = Commander(3)
    # cmd = Commander(4)
    flag = False # So we can print info the FIRST time the loop waits for the vehicles
    # cmd.set_ready(False)
    while not rospy.is_shutdown():
        if not flag:
            print("--- Waiting for robots to indicate READY ---")
            flag = True
        if all(val == True for val in cmd.flag_vals.values()):
            print("*---* All robots have indicated READY *---*")
            cmd.set_ready(True)
            flag = False
        cmd.set_ready(False)



































# from std_msgs.msg import Bool

# import sys
# import rospy
# import numpy as np

# def make_user_wait(msg="Enter exit to exit"):
#     data = raw_input(msg + "\n")
#     if data == 'exit':
#         sys.exit()

# def wait_for_time():
#     """Wait for simulated time to begin """
#     while rospy.Time().now().to_sec() == 0:
#         pass

# if __name__ == "__main__":
#     rospy.init_node('Commander', anonymous=True)
#     wait_for_time()
#     pub = rospy.Publisher('/ready_start_cmd', Bool, queue_size=1)
#     rospy.sleep(0.2)

#     while not rospy.is_shutdown():
#     	val = input("Enter True or False: ")
#     	pub.publish(val)


