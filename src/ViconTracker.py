#! /usr/bin/env/ python2

from geometry_msgs.msg import TransformStamped

# import sys
import rospy
# import numpy as np
# import time


class Tracker(object):
    def __init__(self):
        # self.pub = rospy.Publisher('/ready_start_cmd', Bool, queue_size=1)
        self.data = [None, None, None, None]
        self.yaw = None

        sub3 = rospy.Subscriber('/vicon/TB0/TB0', TransformStamped, self.viconCB0)
        sub1 = rospy.Subscriber('/vicon/TB1/TB1', TransformStamped, self.viconCB1)
        sub2 = rospy.Subscriber('/vicon/TB2/TB2', TransformStamped, self.viconCB2)
        sub3 = rospy.Subscriber('/vicon/TB3/TB3', TransformStamped, self.viconCB3)

    def viconCB0(self, msg):
        self.data[0] = msg.transform

    def viconCB1(self, msg):
        self.data[1] = msg.transform

    def viconCB2(self, msg):
        self.data[2] = msg.transform

    def viconCB3(self, msg):
        self.data[3] = msg.transform


def make_user_wait(msg="Enter exit to exit"):
    data = raw_input(msg + "\n")
    if data == 'exit':
        sys.exit()

def wait_for_time():
    """Wait for simulated time to begin """
    while rospy.Time().now().to_sec() == 0:
        pass

if __name__ == "__main__":
    rospy.init_node('test_vicon', anonymous=True)
    wait_for_time()
    tr = Tracker()
    rospy.sleep(2)
    # rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        # print('TB0 translation is:\n{}'.format(tr.data[0].rotation))
        # print('TB0 translation is:\n{}'.format(tr.data[0].translation))
        # print('TB1 translation is:\n{}'.format(tr.data[1].translation))
        # print('TB2 translation is:\n{}'.format(tr.data[2].translation))
        # print('TB3 translation is:\n{}'.format(tr.data[3].translation))
        # make_user_wait()
        # rate.sleep()




