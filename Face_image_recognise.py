# import the required libraries
import cv2
import pickle

# Load the image and convert it to grayscale
image = cv2.imread('data\Train\Omkar\Omkar.1.8.jpg')
resized = cv2.resize(image,(450,450))
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Load the face cascade classifier
cascade = cv2.CascadeClassifier("./Face-model/haarcascade_frontalface_alt2.xml")

# Load the face recognizer and the trained data
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./Model/face-trainner.yml")

# Load the labels dictionary
with open("./Pickle/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    print(labels)

# Detect faces in the image and make predictions
faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7)
for x, y, w, h in faces:
    face_roi = gray[y:y+h, x:x+w]
    label_id, conf = recognizer.predict(face_roi)
    if conf >= 20 and conf <= 100:
        label = labels[label_id]
        print(f"Label: {label}, Confidence: {conf}")
        cv2.putText(resized, label, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (18,5,255), 2, cv2.LINE_AA)
    else:
        print(conf)
        label = "Unknown"
        cv2.putText(resized, label, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (18,5,255), 2, cv2.LINE_AA)
    resized = cv2.rectangle(resized, (x, y), (x+w, y+h), (0,255,255), 4)

# Show the annotated image
cv2.imshow("Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
