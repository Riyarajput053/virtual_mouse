import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)  
mpDraw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

prev_x, prev_y = None, None
click_time = 0
smooth_factor = 0.2
calibration_points = [(100, 100), (600, 400)] 
while True:
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue 
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

                # if prev_x is None or prev_y is None:
                #     prev_x, prev_y = x1, y1
                # else:
                #     x1 = int((1 - smooth_factor) * prev_x + smooth_factor * x1)
                #     y1 = int((1 - smooth_factor) * prev_y + smooth_factor * y1)
                x_screen = np.interp(x1, (calibration_points[0][0], calibration_points[1][0]), (0, screen_w))
                y_screen = np.interp(y1, (calibration_points[0][1], calibration_points[1][1]), (0, screen_h))

                pyautogui.moveTo(screen_w - x_screen, y_screen, duration=0.04) 

                x2, y2 = lmList[4][1], lmList[4][2] 
                x5 , y5 = lmList[8][1], lmList[8][2]
                index_thumb_dist = math.hypot(x2 - x5, y2 - y5)

                if index_thumb_dist < 30 and time.time() - click_time > 0.3:
                    pyautogui.leftClick()
                    click_time = time.time()

                x3, y3 = lmList[12][1], lmList[12][2] 

                middle_thumb_dist = math.hypot(x2 - x3, y2 - y3)

                if middle_thumb_dist < 30 and time.time() - click_time > 0.3: 
                    pyautogui.rightClick()
                    click_time = time.time()

                y4 = lmList[12][2]  
                scroll_distance = y4 - y1
                if abs(scroll_distance) > 50: 
                    if scroll_distance > 0:
                        pyautogui.scroll(-30) 
                    else:
                        pyautogui.scroll(30) 
                prev_x, prev_y = x1, y1 
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) 
    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()