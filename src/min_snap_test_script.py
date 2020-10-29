#! /usr/bin/env/ python2
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, PoseStamped
from tf.transformations import euler_from_quaternion
from utils_functions import generate_traj_2d

class VelocityController:
    """Simple velocity controller meant to be used with turtlebot3"""
    def __init__(self, odom_topic_name, cmd_vel_topic_name, freq, debug=False):
        self.debug = debug
        self.__odom_sub = rospy.Subscriber(odom_topic_name, Odometry, self.__odomCB)
        self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic_name, Twist, queue_size = 1)

        self.xv = None
        self.yv = None
        self.yaw = None
        self.r = rospy.Rate(freq)
        self.vel_cmd = Twist()

    def __odomCB(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

    def execute_traj(self, xv_list, yv_list):
        # inputs are lists of velocities for x and y
        self.xv = xv_list
        self.yv = yv_list

        print("Starting to head towards the waypoint")

        while True:

            self.cmd_vel_pub.publish(self.vel_cmd)
            self.r.sleep()

        # Stop motion
        self.cmd_vel_pub.publish(Twist())
        if self.debug:
            print("Stopping PID")
            print("Position is currently: ({:.5f},{:.5f})    Yaw is currently: [{:.5f}]".format(self.x, self.y, self.yaw))

        print("** Final Waypoint Reached **")

def wait_for_time():
    """Wait for simulated time to begin """
    while rospy.Time().now().to_sec() == 0:
        pass






if __name__ == "__main__":
    # Initialize node.
    rospy.init_node('testing_min_snap', anonymous=True)
    wait_for_time()

    x_traj = [0, 2, 3]
    y_traj = [0, 1, 3]
    time_full_traj = 30
    freq = 20

    traj_list = generate_traj_2d(x=x_traj, y=y_traj, traj_time=[0,time_full_traj], corr=None, freq=freq)

    while not rospy.is_shutdown():

