import cv2
import mediapipe as mp
import time

class handDetector:
    def __init__(self, static_mode=False, maxhands=2, detection_confident= 0.5, tracking_confident=0.5):
        self.static_mode = static_mode
        self.maxhands = maxhands
        self.detection_confident = detection_confident
        self.tracking_confident = tracking_confident
        
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.static_mode, self.maxhands, self.detection_confident, self.tracking_confident) 
        self.mpdraw = mp.solutions.drawing_utils
        self.top_idx = [4,8,12,16,20]

        
        
    def findhands(self, frame, draw_landmark=True):
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw_landmark:
                    self.mpdraw.draw_landmarks(frame, handlms, self.mphands.HAND_CONNECTIONS)
        
        return frame
    
    def gethandlocation(self, frame, handNo=0, draw_landmark=True):
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo] 
            
            for idx,lm in enumerate(myhand.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                
                self.lmlist.append([idx,cx,cy])
                if draw_landmark:
                    cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)

        return self.lmlist
    
    def fingercheck(self):
        fingers = []
        
        if self.lmlist[self.top_idx[0]][1] > self.lmlist[self.top_idx[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        for idx in range(1,5):
            if self.lmlist[self.top_idx[idx]][2] < self.lmlist[self.top_idx[idx]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

