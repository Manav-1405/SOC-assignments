import wmi
import math
import numpy as np
import cv2

def brightnessControl(img, lmList):
    c = wmi.WMI(namespace='wmi')
    if len(lmList) != 0:
        xa, ya = lmList[4][1], lmList[4][2]
        xb, yb = lmList[8][1], lmList[8][2]
        cv2.circle(img, (xa,ya), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img, (xb,yb), 10, (255,0,255), cv2.FILLED)
        cv2.line(img, (xa,ya), (xb,yb), (255,0,255), 3)

        distance = math.dist((xa,ya),(xb,yb))
        brightness = int(np.interp(distance, [10, 300], [0, 100]))
        c.WmiMonitorBrightnessMethods()[0].WmiSetBrightness(brightness, 0)
        cv2.putText(img, f'Brightness: {brightness}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    return img
