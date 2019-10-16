
# http://169.254.173.218/mjpg/video.mjpg

import numpy as np
import cv2
from datetime import datetime


cap = cv2.VideoCapture()
cap.open("http://root:kamera@169.254.173.218/mjpg/video.mjpg")

while(True):
     # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        timestamp = datetime.timestamp(datetime.now())
        cv2.imwrite('./ircam_data/golfcart/'+str(timestamp)+'.png',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
