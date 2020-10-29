#! /usr/bin/env/ python2

from std_msgs.msg import Bool

import sys
import rospy
import numpy as np

def make_user_wait(msg="Enter exit to exit"):
    data = raw_input(msg + "\n")
    if data == 'exit':
        sys.exit()

def wait_for_time():
    """Wait for simulated time to begin """
    while rospy.Time().now().to_sec() == 0:
        pass

if __name__ == "__main__":
    rospy.init_node('Commander', anonymous=True)
    wait_for_time()
    pub = rospy.Publisher('/ready_start_cmd', Bool, queue_size=1)
    rospy.sleep(0.2)

    while not rospy.is_shutdown():
    	val = input("Enter True or False: ")
    	pub.publish(val)


