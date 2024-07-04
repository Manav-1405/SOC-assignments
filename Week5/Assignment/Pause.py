import cv2
import time
import HandTrakingModule as htm

############################
wCam, hCam = 640, 480
############################

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
        fingerList = detector.fingerUp(lmList)
        if sum(fingerList)>3:
            pass
        else:
            cv2.waitKey(1)
    else:
        cv2.waitKey(1)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("img", img)