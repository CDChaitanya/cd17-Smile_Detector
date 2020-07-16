# -*- coding: utf-8 -*-

# Smile Detector
import cv2

# LOADING THE CASCADES
face_cascade  = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   #FOR FACE
eye_cascade   = cv2.CascadeClassifier('haarcascade_eye.xml')                   #FOR EYE
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')                 #FOR SMILE

# DEFINING THE FUNCTION THAT WILL DO DETECTION
def detect(gray , frame):
    faces = face_cascade.detectMultiScale(image=gray ,scaleFactor=1.3  , minNeighbors=5 )
    for (x , y , w , h) in faces:
        cv2.rectangle(img=frame , pt1= (x,y) , pt2= (x+w , y+h), color=(255, 0, 0),thickness=2)

        roi_gray  = gray [y:y+h , x:x+w]             #REGION OF INTREST(ROI)
        roi_frame = frame[y:y+h , x:x+w]

        eyes = eye_cascade.detectMultiScale(image=roi_gray ,scaleFactor=1.1  , minNeighbors= 22)
        for (ex , ey , ew , eh) in eyes:
            cv2.rectangle(img=roi_frame , pt1= (ex,ey) , pt2= (ex+ew , ey+eh), color=(0,255,0),thickness=2)

        smiles = smile_cascade.detectMultiScale(image=roi_gray,scaleFactor=1.1, minNeighbors= 22)
        for (sx , sy , sw , sh) in smiles:
            cv2.rectangle(img=roi_frame , pt1= (sx,sy) , pt2= (sx+sw , sy+sh), color=(0,0,255),thickness=2)

    return frame

#DOING SOME FACE RECOGNITION WITH WEBCAM
video_capture = cv2.VideoCapture(0)                  # (0=internal webcam / 1= external webcam)
while True:
    _, frame = video_capture.read()                  # capturing the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # converting frame to gray 
    canvas = detect(gray, frame)                     # canvas to print it on webcam 
    cv2.imshow('Video' , canvas)                     # printing on video window
    if cv2.waitKey(1) & 0xFF==ord('q') :
        break                                        # 'q' to exit the video window
    
video_capture.release() 
cv2.destroyAllWindows()