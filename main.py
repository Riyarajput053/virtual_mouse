import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
   

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            if lmList:
                x1, y1 = lmList[8][1], lmList[8][2]
                x_screen = np.interp(x1, (100, 600), (0, screen_w))
                y_screen = np.interp(y1, (100, 400), (0, screen_h))
                pyautogui.moveTo(screen_w - x_screen, y_screen, duration=0.3)

                x2, y2 = lmList[4][1], lmList[4][2]
                index_thumb_dist = math.hypot(x2 - x1, y2 - y1)

                if index_thumb_dist < 40:
                    pyautogui.click()
                    
                    
                x3, y3 = lmList[12][1], lmList[12][2]
                middle_thumb_dist = math.hypot(x2-x3, y2-y3)
                    
                if middle_thumb_dist< 40:
                    pyautogui.rightClick()
                    
                y4 = lmList[12][2]  
                scroll_distance = y4 - y1  
                if abs(scroll_distance) > 20:  
                    if scroll_distance > 0:  
                        pyautogui.scroll(-40)  
                    else:  
                        pyautogui.scroll(40)  
                

               

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
