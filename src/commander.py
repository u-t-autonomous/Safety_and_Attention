#! /usr/bin/env/ python2

import rospy
import command_control as cc


if __name__ == "__main__":
    rospy.init_node('robot_command', anonymous=True)
    #cmd = Commander(3)
    cmd = cc.Commander(0)
    flag = False # So we can print info the FIRST time the loop waits for the vehicles
    # cmd.set_ready(False)
    while not rospy.is_shutdown():
        print(cmd.flag_vals.values())
        if not flag:
            print("--- Waiting for robots to indicate READY ---")
            flag = True
        if all(val == True for val in cmd.flag_vals.values()):
            print("*---* All robots have indicated READY *---*")
            cmd.set_ready(True)
            flag = False
        cmd.set_ready(False)
