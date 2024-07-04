import cv2
import time
import HandTrakingModule as htm

############################
wCam, hCam = 640, 480
############################

detector = htm.handDetector(detectionCon=0.7)

image_path = 'Fingers/0.png'
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

dragging = False
offset_x = 0
offset_y = 0
image_position = (100, 100)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
ptime = 0

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = detector.fingerUp(lmList)
        if fingers[1] == 1 and all(f == 0 for f in fingers[2:]):
            index_finger_x, index_finger_y = lmList[8][1], lmList[8][2]

            if not dragging:
                if image_position[0] < index_finger_x < image_position[0] + image_width and image_position[1] < index_finger_y < image_position[1] + image_height:
                    dragging = True
                    offset_x = image_position[0] - index_finger_x
                    offset_y = image_position[1] - index_finger_y
            else:
                image_position = (index_finger_x + offset_x, index_finger_y + offset_y)
        else:
            dragging = False

    x, y = image_position
    img[y:y + image_height, x:x + image_width] = image

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)