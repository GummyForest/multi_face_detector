# DETECT SEVERAL PHOTOS FACES

import cv2
import glob  # glob check all directory for files

gimg = glob.glob("*.jpg")  # scan for every .jpg files (glob image - gimg)

detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# importing classifier

for timg in gimg:  # (travesse image - timg)
    img = cv2.imread(timg)  # reading the current image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting to gray scale

    faces = detect.detectMultiScale(gray, 1.2, 5)  # detecting faces

    for (x, y, w, h) in faces:  # drawing retangles
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detector", img)  # opening window
    cv2.waitKey(2000)  # time window open
    cv2.destroyAllWindows()   # close window
