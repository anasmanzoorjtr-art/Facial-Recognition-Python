
import cv2
import os
import glob
import json

# Dataset folder
dataset_path = "dataset"

# Face detection model
cascade_path = "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

# LBPH recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}

label_id = 0
image_paths = glob.glob(dataset_path + "/**/*.jpg", recursive=True)
image_paths += glob.glob(dataset_path + "/**/*.png", recursive=True)

print("Starting Face Training...")

for image_path in image_paths:

    # Person name is the folder immediately inside dataset
    parts = os.path.normpath(image_path).split(os.sep)
    dataset_index = parts.index("dataset")
    person_name = parts[dataset_index + 1]

    if person_name not in label_names:
        label_names[person_name] = label_id
        label_id += 1

    image = cv2.imread(image_path)

    if image is None:
        print("Cannot read:", image_path)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected_faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(detected_faces) == 0:
        print("No face detected:", image_path)
        continue

    # Use the largest detected face
    x, y, w, h = max(detected_faces, key=lambda box: box[2] * box[3])

    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200))

    faces.append(face)
    labels.append(label_names[person_name])

    print("Processed:", os.path.basename(image_path))

if len(faces) < 2:
    print("ERROR: Not enough face images found.")
    print("Please check your dataset images.")
else:
    recognizer.train(faces, __import__("numpy").array(labels))
    recognizer.save("face_model.yml")

    with open("labels.json", "w") as file:
        json.dump(label_names, file)

    print("\nTRAINING COMPLETED SUCCESSFULLY!")
    print("Images used:", len(faces))
    print("People:", list(label_names.keys()))
    print("Model saved: face_model.yml")
