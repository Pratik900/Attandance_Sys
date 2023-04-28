# import the required libraries
import cv2
import pickle
from xlsxwriter import Workbook
import pandas as pd
from datetime import date,datetime

video = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("./Face-Model/haarcascade_frontalface_alt2.xml")

# Loaading the face recogniser and the trained data into the program
recognise = cv2.face.LBPHFaceRecognizer_create()
recognise.read("./Model/face-trainner.yml")

labels = {} # dictionary
labels_count = 0
# Opening labels.pickle file and creating a dictionary containing the label ID
# and the name
with open("./Pickle/face-labels.pickle", 'rb') as f:##
    og_label = pickle.load(f)##
    labels = {v:k for k,v in og_label.items()}##
    print(labels)


# Initialize dictionaries to keep track of the number of times each ID and name are detected
id_count = {}
name_count = {}

while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = cascade.detectMultiScale(gray, scaleFactor=1.9, minNeighbors=5)
    for x, y, w, h in face:
        face_save = gray[y:y+h, x:x+w]
        ID, conf = recognise.predict(face_save)
        name = labels[ID]

        if conf >= 20 and conf <= 100:
            cv2.putText(frame, name, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (18, 5, 255), 2, cv2.LINE_AA)
            time_now = datetime.now()
            current_time = time_now.strftime("%H:%M:%S")
            df_old = pd.read_excel("attendance.xlsx", sheet_name="attendance")

            # Count the number of times each ID and name are detected
            if ID in id_count:
                id_count[ID] += 1
            else:
                id_count[ID] = 1

            print(id_count[ID])
            if id_count[ID] == 5 and str(labels[ID]) not in df_old['Name'].values:
                dataframe = pd.DataFrame({'Name': [str(labels[ID])],
                                           'Date': [str(date.today())],
                                           'Time': [str(current_time)]
                                          })
                dataframe = df_old.append(dataframe)
                writer = pd.ExcelWriter("attendance.xlsx", engine='xlsxwriter')
                dataframe.to_excel(writer, sheet_name="attendance", index=False)
                writer.save()

                id_count[ID] = 0
            if id_count[ID] == 5:
                id_count[ID] = 0

        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 4)

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
