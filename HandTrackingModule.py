import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode = False, maxHands = 1, detectionCon = 0.5,trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode= self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence= self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self,img,draw = True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        # imgRGB = img
        self.results = self.hands.process(imgRGB)
        # print(dir(self.results))
        # print(self.results.multi_hand_world_landmarks)
        if self.results.multi_hand_landmarks:
            for handLM in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLM,self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self,img, handNo = 0):
        lmList = []
        min_x, min_y = 1e9, 1e9
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                # cx, cy = int(lm.x * w), int(lm.y * h)
                # lmList.append([id,cx,cy])
                # cx, cy = int(lm.x * w), int(lm.y * h)
                min_x = min(min_x, lm.x)
                min_y = min(min_y, lm.y)
                lmList.extend([lm.x, lm.y])
        
        for i in range(0, len(lmList), 2):
            lmList[i] -= min_x
            lmList[i + 1] -= min_y
        return lmList

    
def main():
    pTime = 0
    cap = cv.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        try:
            fingers = detector.getFingers(img)
            print(fingers)
        except Exception as ex:
            print(f'An Exception Occurred: {ex}')
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow('image', img)
        k = cv.waitKey(1)
        if k == 27:
            cv.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
