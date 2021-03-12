#! /usr/bin/env/ python2

# Imports for ROS side
import rospy
import numpy as np
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from tf.transformations import euler_from_quaternion
from Safety_and_Attention.msg import Ready
import command_control as cc
# Imports for Algorithm side
import random


# To import for running safety and attention simulations
# from linear_dynamics_auxilliary_functions import (ellipsoid_level_function,
#     check_collisions,continue_condition,get_mu_sig_over_time_hor,
#     get_mu_sig_theta_over_time_hor,get_Q_mats_over_time_hor,
#     get_Qplus_mats_over_time_hor,propagate_linear_obstacles,
#     propagate_nonlinear_obstacles)
# from dc_based_motion_planner import (create_dc_motion_planner,
#     solve_dc_motion_planning,create_cvxpy_motion_planner_obstacle_free)
# from dc_motion_planning_ecos import (dc_motion_planning_call_ecos,
#     stack_params_for_ecos,set_up_independent_ecos_params)
# from Robot_SaA_Environment import LinearObstacle, NonlinearObstacle, RobotSaAEnvironment

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

    def go_to_point(self, goal):

        # input variable goal should be of type geometry_msgs/Point

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
            print("Position is currently: ({:.5f},{:.5f})    Yaw is currently: [{:.5f}]".format(self.x, self.y, self.yaw))

        print("** Waypoint Reached **")

# ------------------ End Class Definitions --------------------

# ------------------ Start Function Definitions ---------------
def make_user_wait(msg="Enter exit to exit"):
    data = raw_input(msg + "\n")
    if data == 'exit':
        sys.exit()

def wait_for_time():
    """Wait for simulated time to begin """
    while rospy.Time().now().to_sec() == 0:
        pass
# ------------------ End Function Definitions -----------------

if __name__ == '__main__':

    np.random.seed(0)

    rospy.init_node("robot_control_3_node", anonymous=True)
    wait_for_time()

    # Create velocity controller and converter objects
    vel_controller_3 = VelocityController('/tb3_3/odom', '/tb3_3/cmd_vel')
    robot_name='tb3_3'
    rospy.sleep(1.0)

    # Set the initial point of the robotic agent in the Gazebo world (make sure this
    # is the same as the initial position in the Safety and Attention environment)
    init_point_3 = Point(2.25, -2.0, None)
    vel_controller_3.go_to_point(init_point_3)

    # Dynamics of third obstacle
    sampling_time = 1.
    obs_3_A_matrix = np.eye(2)
    obs_3_F_matrix = sampling_time*np.eye(2)
    obs_3_mean_vec = np.array([0.025, 0.15])
    obs_3_cov_mat = np.array([[0.005, 0.0015], [0.0015, 0.008]])

    # Generate a set of waypoints for the first obstacle to follow
    num_steps = 200
    traj = []
    traj_np = []
    for step in range(0,num_steps):
        if step == 0:
            # Query the previous point in the set of waypoints
            prev_x = init_point_3.x
            prev_y = init_point_3.y
            prev_state = np.array([prev_x,prev_y])
            # Push this point through the dynamics
            obs_w_step = np.random.multivariate_normal(obs_3_mean_vec,obs_3_cov_mat,1)
            obs_w_step = np.reshape(obs_w_step, (2,))
            new_state = np.matmul(obs_3_A_matrix,prev_state) + np.matmul(obs_3_F_matrix,obs_w_step)
            new_point = Point(float(new_state[0]),float(new_state[1]), None)
            traj = [new_point]
            traj_np = [np.array(new_state)]
        else:
            # Query the previous point in the set of waypoints
            prev_x = traj[step-1].x
            prev_y = traj[step-1].y
            prev_state = np.array([prev_x, prev_y])
            # Push this point through the dynamics
            obs_w_step = np.random.multivariate_normal(obs_3_mean_vec, obs_3_cov_mat, 1)
            obs_w_step = np.reshape(obs_w_step, (2,))
            new_state = np.matmul(obs_3_A_matrix, prev_state) + np.matmul(obs_3_F_matrix, obs_w_step)
            new_point = Point(float(new_state[0]), float(new_state[1]), None)
            traj.append(new_point)
            traj_np.append(np.array(new_state))

    np.save("obstacle_3_trajectory", np.array(traj_np))

    # Wait until all other robots are ready
    rdy = cc.ReadyTool(robot_name)
    print("*** Robot {} is ready and waiting to start ***".format(int(robot_name[-1])))
    rdy.set_ready(True)
    rdy.wait_for_ready()
    # print("Robot {} made it past Ready Check *".format(int(robot_name[-1]))) # Comment when done testing
    # sys.exit() # Comment when done testing

    #traj = np.load('obs_3_states.npy')

    # Now, while we have not reached the target point, continue executing the controller
    while not rospy.is_shutdown():
        for next_p in traj_np:
            rdy.set_ready(False)
            next_x = next_p[0]
            next_y = next_p[1]
            next_point = Point(next_x, next_y, None)
            vel_controller_3.go_to_point(next_point)
            rdy.set_ready(True)
            # Wait for the agent and the obstacles to have synchronized to their next state
            rdy.wait_for_ready()
            print("Robot {} is moving to the next waypoint *".format(int(robot_name[-1])))

