import math
import numpy as np
import cv2
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



def soundControl(img, lmList):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)

    volRange = volume.GetVolumeRange()
    volMin = volRange[0]
    volMax = volRange[1]
    if len(lmList) != 0:
        xa, ya = lmList[4][1], lmList[4][2]
        xb, yb = lmList[8][1], lmList[8][2]
        cv2.circle(img, (xa,ya), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img, (xb,yb), 10, (255,0,255), cv2.FILLED)
        cv2.line(img, (xa,ya), (xb,yb), (255,0,255), 3)

        distance = math.dist((xa,ya),(xb,yb))
        vol = int(np.interp(distance, [10, 300], [volMin, volMax]))
        volume.SetMasterVolumeLevel(vol, None)
        cv2.putText(img, f'Volume: {vol}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    return img
