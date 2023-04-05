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


while True:
    check,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = cascade.detectMultiScale(gray, scaleFactor = 1.9, minNeighbors = 5)
    #print(face)
    for x,y,w,h in face:
        face_save = gray[y:y+h, x:x+w]
        # Predicting the face identified
        ID, conf = recognise.predict(face_save)
        #print(ID,conf)
        if conf >= 20 and conf <= 100:
            print(conf)
            print(ID)
            print(labels[ID])
            cv2.putText(frame,labels[ID],(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX ,1, (18,5,255), 2, cv2.LINE_AA ) 

            time_now = datetime.now()
            current_time = time_now.strftime("%H:%M:%S")
            df_old = pd.read_excel("attendance.xlsx",sheet_name="attendance")
            dataframe=pd.DataFrame({'Name':[str(labels[ID])],
                                    'Date':[str(date.today())],
                                    'Time':[str(current_time)]
                                    })
            
            dataframe=df_old.append(dataframe)
            writer = pd.ExcelWriter("attendance.xlsx", engine='xlsxwriter')
            dataframe.to_excel(writer,sheet_name = "attendance", index=False)
            writer.save()


        frame = cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,255),4)

    cv2.imshow("Video",frame)
    key = cv2.waitKey(1)
    if(key == ord('q')):
        break

video.release()
cv2.destroyAllWindows()
