# start by importing opencv-python module
import cv2

# for face detection the pre-trained haarcascades classifier is loaded
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# the webcam is then initialize depending on how many cameras
cap = cv2.VideoCapture(0)

while True:
    # a frame is read from the webcam
    ret, frame = cap.read()

    # the frame is converted to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces is detected in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # this would draw rectangle around the detected faces with red color
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # this would display the resulting frame
    cv2.imshow('face Detection', frame)

    # 'k' key is pressed to BREAK the loop
    if cv2.waitKey(1) & 0xFF == ord('k'):
        break

# the webcam is then released and all OpenCV windows is closed
cap.release()
cv2.destroyAllWindows()
