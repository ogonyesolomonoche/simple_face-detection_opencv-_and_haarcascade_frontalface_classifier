# start by importing opencv-python module
import cv2

def detect_faces(image_path, cascade_path):
    # the cascade classifier is then loaded
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # the input image is read by opencv
    image = cv2.imread(image_path)

    # the image is converted to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # the face detection is done  using the  pre_trained cascade classifier
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # this would a blue rectangle around the detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # the output image is then display
    cv2.imshow('face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # loaded path to the input image and Haar Cascade XML file
    input_image_path = "test_img.jfif"
    cascade_xml_path = "haarcascade_frontalface_default.xml"

    # face detection is then performed
    detect_faces(input_image_path, cascade_xml_path)
