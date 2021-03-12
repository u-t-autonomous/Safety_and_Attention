#! /usr/bin/env/ python2

''' Test file to make sure object detection is working '''

import rospy
from detect_colors import object_map

def wait_for_time():
    """Wait for simulated time to begin """
    while rospy.Time().now().to_sec() == 0:
        pass

def make_user_wait(msg="Enter exit to exit"):
    data = raw_input(msg + "\n")
    if data == 'exit':
        sys.exit()

if __name__ == '__main__':
    rospy.init_node("robot_control_0_node", anonymous=True)
    wait_for_time()

    obstacle_tracker = object_map()
    while(1):
    	obs_in_view = obstacle_tracker.which_obj()
    	print(obs_in_view)
    	make_user_wait()
