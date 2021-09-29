#!/usr/bin/env python

# Class for subscribing to a ROS image topic and getting a mask

import rospy
import cv2
import numpy as np
import sys
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class object_map(object):

    def __init__(self):
        self.image_sub = rospy.Subscriber("/tb3_0/camera/rgb/image_raw", Image, self.image_CB)
        self.bridge = CvBridge()
        # self.visualizer = rospy.Publisher("/color_id_mask", Image, queue_size=10)
        self.height = None
        self.width = None
        self.object_ID = None

    def which_obj(self):
        return self.object_ID

    def image_CB(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        try:
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except Exception as e:
            print(e)
        self.height, self.width = hsv_image.shape[:2]
        # Ranges for each color
        # red =   ([136, 87, 111] ,[200, 255, 255], 'red')
        red =   ([0, 255, 99]   ,[0, 255, 107]  , 'red')
        green = ([25, 52, 72]   ,[102, 255, 255], 'green')
        blue =  ([94, 80, 2]    ,[120, 255, 255], 'blue')
        # List colors to look for in colors
        colors = [green, blue, red]
        output = cv_image
        # kernal = np.ones((self.height, self.width), "uint8")
        kernal = np.ones((5, 5), "uint8")
        self.object_ID = []

        for color in colors:
            # create NumPy arrays from the boundaries
            lower = np.array(color[0], np.uint8)
            upper = np.array(color[1], np.uint8)
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(hsv_image, lower, upper)
            mask = cv2.dilate(mask, kernal)
            res = cv2.bitwise_and(cv_image, cv_image, mask = mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area > 50):
                    self.object_ID.append(color[2])
                    x, y, w, h = cv2.boundingRect(contour)
                    output = cv2.rectangle(output, (x,y),
                                            (x + w, y + h),
                                            (255, 0, 0), 2)
                    cv2.putText(output, color[2], (x,y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 0))

        cv2.imshow('Color Detection', output)
        cv2.waitKey(1)
        # try:
        #     self.visualizer.publish(self.bridge.cv2_to_imgmsg(output, "bgr8"))
        # except CvBridgeError as e:
        #     print(e)

        
def main(args):
    '''Initializes and cleanup ROS node'''
    rospy.init_node('identify_color', anonymous=True)
    slam = object_map()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down color id")

if __name__ == '__main__':
    main(sys.argv)
