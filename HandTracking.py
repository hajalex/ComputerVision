import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    x, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # The location of each landmark
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(f'id:{id} --> {cx} {cy}')
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv.imshow("image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
