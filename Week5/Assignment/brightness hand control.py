import cv2
import time
import numpy as np
import HandTrakingModule as htm
import wmi
import math

############################
wCam, hCam = 640, 480
############################

c = wmi.WMI(namespace='wmi')
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
ptime = 0

detector = htm.handDetector(detectionCon=0.5)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 10, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)

        distance = math.dist((x1,y1),(x2,y2))
        brightness = int(np.interp(distance, [10, 300], [0, 100]))
        c.WmiMonitorBrightnessMethods()[0].WmiSetBrightness(brightness, 0)
        cv2.putText(img, f'Brightness: {brightness}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

    cv2.imshow("img", img)
    cv2.waitKey(1)