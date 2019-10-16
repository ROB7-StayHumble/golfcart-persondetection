#!/usr/bin/env python2
# http://169.254.173.218/mjpg/video.mjpg

import rospy
import numpy as np
import cv2
from datetime import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

cap = cv2.VideoCapture()
cap.open("http://root:kamera@169.254.173.218/mjpg/video.mjpg")

rospy.init_node('ircam_data', anonymous=True)
bridge = CvBridge()
image_pub = rospy.Publisher("/ircam_data",Image)

while(True):
     # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.imshow('IRcam',frame)
    
    image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))

    if cv2.waitKey(1) & 0xFF == ord('c'):
        print("capturing file")
        timestamp = datetime.timestamp(datetime.now())
        cv2.imwrite('./ircam_data/'+str(timestamp)+'.png',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
