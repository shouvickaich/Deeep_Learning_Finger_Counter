import cv2
import mediapipe as mp
import time

# speed is inversely propportional to accuracy
# modelCompexity : 0,1,2
# 0- fast , not accurate
# 1 - medium , good enough results
# 2- slow , accurate

# Confidence : 0-1
# detectionCon -> if it is a hand
# trackcon -> inside hand 20 point


class HandDetector():
    def __init__(self, mode=False, maxHands=2,modelComplexity=1, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Detecting hand , detecting points (20), drawing points
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        # 20 points -> x and y coordinates of 20 points
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList
    
# laptops: 640*480 ,400*600 

# converting model size: 320*480

# 0 point -> x,y
# 1 point ->x,y

# ....
# 20 points