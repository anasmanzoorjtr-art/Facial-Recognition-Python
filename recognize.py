
import cv2
import json

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

with open("labels.json", "r") as file:
    labels = {int(v): k for k, v in json.load(file).items()}

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image = cv2.imread("test.jpg")

if image is None:
    print("ERROR: test.jpg not found!")
    raise SystemExit

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Enlarge image to help face detection
gray = cv2.resize(gray, None, fx=1.5, fy=1.5)

faces = detector.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=3,
    minSize=(30, 30)
)

if len(faces) == 0:
    print("No face detected. Use a clear front-facing photo.")
else:
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label_id, confidence = recognizer.predict(face)

        name = labels.get(label_id, "Unknown")

        if confidence >= 90:
            name = "Unknown"

        print("Result:", name)
        print("Confidence:", round(confidence, 2))

        cv2.rectangle(image, (int(x/1.5), int(y/1.5)),
                      (int((x+w)/1.5), int((y+h)/1.5)), (0, 255, 0), 2)

        cv2.putText(image, name, (int(x/1.5), int(y/1.5)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite("recognized_output.jpg", image)
    print("Saved: recognized_output.jpg")
