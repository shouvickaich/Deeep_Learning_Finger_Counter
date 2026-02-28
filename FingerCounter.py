import cv2
import numpy as np
import math
import HandTrackingModule as htm
import time
import os

# if frame processing takes long time-> 1 second less frames would be run -> less fps
# if model is fast -> more frames would be run -> more fps


# Open Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # height

folderPath = "Finger-Counter/images"
imageList = os.listdir(folderPath)
print(imageList)
imageList.sort()
print("Sorted Image List:", imageList)
overlayList = []
for imPath in imageList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))

pTime = 0 #-> prcessed frame time
cTime = 0 #-> current frame time

detector = htm.HandDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    # Read Frame
    success, img = cap.read()
    # Flip Image
    img = cv2.flip(img, 1)
    # Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        fingers = []
        for id in range(0, 5):
            if id !=0:
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                
                if lmList[tipIds[id]][1] < lmList[tipIds[id] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
     # [1,0,0,0,0]
        totalFingers = fingers.count(1)
        print(fingers)
        
        h, w, c = overlayList[totalFingers].shape
        img[0:h, 0:w] = overlayList[totalFingers] 
        print("IMAGE: ",h, w, c)
        pos = int(w/2)
        cv2.rectangle(img, (0, h), (w, h+60), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (pos - 20, h+55), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)
    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # rhs
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    # Display Image
    cv2.imshow("Image", img)
    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# id  x  y
# 0 123 340
# 1
# 2
# 3