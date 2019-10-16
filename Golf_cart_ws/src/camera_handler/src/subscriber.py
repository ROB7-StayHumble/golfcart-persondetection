#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class camera_processing():

    def __init__(self):

        rospy.init_node('camera_handler', anonymous=True)
        self.bridge = CvBridge()
        
        rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.callback)
     
    
    def callback(self, img_data):
        try:
            image = self.bridge.imgmsg_to_cv2(img_data, "32FC1")
	    cv2.imshow(image)
        except CvBridgeError, e:
            print e


if __name__ == '__main__': 
    try:
        detector = camera_processing()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Camera_handler node terminated.")
