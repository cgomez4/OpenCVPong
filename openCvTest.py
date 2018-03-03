import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    fist_cascade = cv2.CascadeClassifier('fist.xml')
    closed_frontal_palm_cascade = cv2.CascadeClassifier('closed_frontal_palm.xml')
    palm_cascade = cv2.CascadeClassifier('palm.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    fists = fist_cascade.detectMultiScale(gray, 1.3, 5)
    closed_frontal_palms = closed_frontal_palm_cascade.detectMultiScale(gray, 1.3, 5)
    palms = palm_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w ,h) in fists:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()