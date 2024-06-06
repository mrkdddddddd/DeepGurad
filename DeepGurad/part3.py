import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import tkinter as tk
from tkinter import simpledialog

# 사용자 이름 입력 받는 GUI 생성
root = tk.Tk()
root.withdraw()  # Tkinter 창 숨기기
user_name = simpledialog.askstring("Input", "Enter your name:")

data_path = 'changed_photos/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

models_path = 'models'
if not os.path.exists(models_path):
    os.makedirs(models_path)

model.save(os.path.join(models_path, f"{user_name}_model.xml"))

print("Model Training Complete!!!!!")

# 사용자용

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi

# 모델 로드
models = {}
model_files = [f for f in listdir(models_path) if isfile(join(models_path, f)) and f.endswith('_model.xml')]

for file in model_files:
    model_name = file.replace('_model.xml', '')
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(join(models_path, file))
    models[model_name] = model

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        min_confidence = float('inf')
        best_match = None

        for model_name, model in models.items():
            result = model.predict(face)
            if result[1] < min_confidence:
                min_confidence = result[1]
                best_match = model_name

        if min_confidence < 500:
            confidence = int(100 * (1 - (min_confidence) / 300))
            display_string = f'{confidence}% Confidence it is {best_match}'
        else:
            display_string = "Unknown"

        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face Cropper', image)

    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1) == 13:  # Enter 키
        break

cap.release()
cv2.destroyAllWindows()
