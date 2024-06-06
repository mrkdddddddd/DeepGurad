import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

def create_model():
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_extractor(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None

        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]

        return cropped_face

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            if not os.path.exists('faces'):
                os.makedirs('faces')

            file_name_path = 'faces/user'+str(count)+'.jpg'
            cv2.imwrite(file_name_path, face)

            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face not Found")
            pass

        if cv2.waitKey(1) == 13 or count == 100:
            break

    print('Collecting Samples Complete!!!')

    # Augmentation
    photo_dir = "./faces/"
    changed_photo_dir = "./changed_photos/"
    photo_names = [f for f in os.listdir(photo_dir) if f.endswith('.jpg')]

    if not os.path.exists("changed_photos"):
        os.makedirs("changed_photos")

    def augment_image(image):
        augmented_images = []
        
        augmented_images.append(image)
        
        flipped_image = cv2.flip(image, 1)
        augmented_images.append(flipped_image)
        
        rows, cols, _ = image.shape
        for angle in range(-10, 11, 5):
            M_rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_image = cv2.warpAffine(image, M_rotation, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
            augmented_images.append(rotated_image)
        
        shear_range = 10
        for shear in range(-shear_range, shear_range + 1, 5):
            M_shear = np.float32([[1, shear / 100, 0], [0, 1, 0]])
            sheared_image = cv2.warpAffine(image, M_shear, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
            augmented_images.append(sheared_image)
        
        for brightness in [-50, 50]:
            brightened_image = cv2.convertScaleAbs(image, beta=brightness)
            augmented_images.append(brightened_image)
        
        return augmented_images

    if not os.path.exists('changed_photos'):
        os.makedirs('changed_photos')

    for photo_name in photo_names:
        photo_path = os.path.join(photo_dir, photo_name)
        image = cv2.imread(photo_path)
        
        augmented_images = augment_image(image)
        
        base_name, ext = os.path.splitext(photo_name)
        for idx, augmented_image in enumerate(augmented_images):
            augmented_photo_name = f"{base_name}_augmented_{idx}{ext}"
            augmented_photo_path = os.path.join(changed_photo_dir, augmented_photo_name)
            cv2.imwrite(augmented_photo_path, augmented_image)

    print('Augmentation Complete!!!')
    cap.release()
    cv2.destroyAllWindows()
    
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

