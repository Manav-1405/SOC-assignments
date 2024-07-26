import cv2
import numpy as np
import time
import os
import HandTrakingModule as htm
import brightnessModule as bm
import soundModule as sm

############################
brushThickness = 60
eraserThickness = 120
wCam, hCam = 1280, 720
############################

folderPath = "header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
ptime = 0
draw = False
drawColor = (0,0,0)
soundMode = False
brightnessMode = False

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 720, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]
        fingerList = detector.fingerUp(lmList)
        if fingerList[1]:
            if fingerList[2]:
                xp, yp = 0, 0
                if y1 < 125:
                    if 0<x1<140:
                        header = overlayList[8]
                        draw = False
                        soundMode = False
                        brightnessMode = True
                    if 150<x1<290:
                        header = overlayList[7]
                        draw = False
                        brightnessMode = False
                        soundMode = True
                    if 310<x1<610:
                        header = overlayList[6]
                        soundMode = False
                        brightnessMode = False
                        draw = False
                        os.chdir("D:/coding skills/SOC/Test images/")
                        cv2.imwrite('image.jpg', imgCanvas)
                    if 630<x1<760:
                        header = overlayList[5]
                        soundMode = False
                        brightnessMode = False
                        drawColor = (0,0,0)
                        draw = True
                    if 770<x1<895:
                        header = overlayList[4]
                        soundMode = False
                        brightnessMode = False
                        draw = True
                        drawColor = (0,255,0)
                    if 905<x1<1025:
                        header = overlayList[3]
                        soundMode = False
                        brightnessMode = False
                        draw = True
                        drawColor = (255,0,0)
                    if 1035<x1<1155:
                        header = overlayList[2]
                        soundMode = False
                        brightnessMode = False
                        draw = True
                        drawColor = (0,0,255)
                    if 1165<x1<1280:
                        header = overlayList[1]
                        soundMode = False
                        brightnessMode = False
                        draw = True
                        drawColor = (255,255,255)
                cv2.rectangle(img, (x1, y1-25), (x2, y1+25), drawColor, cv2.FILLED)
            else:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if fingerList[0] and brightnessMode:
                img = bm.brightnessControl(img,lmList)
            if fingerList[0] and soundMode:
                img = sm.soundControl(img,lmList)
        
        if fingerList[1] and fingerList[2] == False and draw:
            cv2.circle(img, (x1, y1), brushThickness, drawColor, cv2.FILLED)
        
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

    header = cv2.resize(header, (1280,125))
    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
