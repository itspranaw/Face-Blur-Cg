import cv2

video = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    success, img = video.read()
    faces = face_cascade.detectMultiScale(img, 1.3, 4)
    for (x, y, w, h) in faces:
        ROI = img[y:y+h, x:x+w]

        blur = cv2.GaussianBlur(ROI, (99, 99), 0)
        img[y:y+h, x:x+w] = blur

    cv2.imshow("Blurred Face", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

video.release()
cv2.destroyAllWindows()