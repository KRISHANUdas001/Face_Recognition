import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x-20, y-20), (x + w+20, y + h+20), (0, 255, 0), 4)
        Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if Id == 1:
            Id = "Krishanu {0:.2f}%".format(round(100 - confidence, 2))
        else:
            Id = "Unknown"
        cv2.rectangle(img, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
        cv2.putText(img, str(Id), (x, y+h), font, 1, (255, 255, 255), 3)
    cv2.imshow('img', img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
