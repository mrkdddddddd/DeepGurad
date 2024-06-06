import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import simpledialog

def create_model():
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_extractor(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        if faces is ():
            return None

        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]

        return cropped_face

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            if not os.path.exists('faces'):
                os.makedirs('faces')

            file_name_path = 'faces/user'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==100:
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

    model.train(np.asarray(Training_Data), np.asarray(Labels))

    models_path = 'models'
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    model.save(os.path.join(models_path, f"{user_name}_model.xml"))

    print("Model Training Complete!!!!!")

class AdminPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Admin Page")
        self.root.geometry("300x300")
        self.manage_window = None  # Manage Models 창을 위한 속성 추가
        
        self.create_button = tk.Button(root, text="Create Model", command=create_model, bg='#404040', fg='black', width=10, height=2, relief="raised", borderwidth=0, highlightthickness=0, highlightbackground='white')
        self.create_button.pack(pady=10)
        self.create_button.config(bg="#D9D9D9", padx=20, border=5)
        self.create_button.pack(pady=[50,0])
        
        self.manage_button = tk.Button(root, text="Manage Models", command=self.manage_models, bg='#404040', fg='black', width=10, height=2, relief="raised", borderwidth=0, highlightthickness=0, highlightbackground='white')
        self.manage_button.pack(pady=10)
        self.manage_button.config(bg="#D9D9D9", padx=20, border=5)
        self.manage_button.pack(pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=self.root.destroy, bg='#404040', fg='black', width=10, height=2, relief="raised", borderwidth=0, highlightthickness=0, highlightbackground='white')
        self.exit_button.pack(pady=10)
        self.exit_button.config(bg="#D9D9D9", padx=20, border=5)
        self.exit_button.pack(pady=[40, 40])

    def manage_models(self):
        models_path = 'models'
        model_files = [f for f in listdir(models_path) if isfile(join(models_path, f)) and f.endswith('_model.xml')]
        
        self.manage_window = tk.Toplevel(self.root)
        self.manage_window.title("Manage Models")
        self.manage_window.geometry("300x400")
        
        self.model_listbox = tk.Listbox(self.manage_window)
        self.model_listbox.pack(expand=True, fill=tk.BOTH)
        for model_file in model_files:
            self.model_listbox.insert(tk.END, model_file)

        self.delete_button = tk.Button(self.manage_window, text="Delete", command=self.delete_model, fg="white")
        self.delete_button.pack(pady=10)
        self.delete_button.config(bg="#f53e05", padx=20, border=5)

        self.exit_button = tk.Button(self.manage_window, text="Exit", command=self.manage_window.destroy)
        self.exit_button.pack(pady=10)
        self.exit_button.config(bg="#D9D9D9", padx=20, border=5)

    def delete_model(self):
        selected_index = self.model_listbox.curselection()
        if selected_index:
            selected_model = self.model_listbox.get(selected_index[0])
            if messagebox.askokcancel("Delete Model", f"Do you want to delete {selected_model}?"):
                os.remove(os.path.join('models', selected_model))
                self.model_listbox.delete(selected_index)


if __name__ == "__main__":
    root = tk.Tk()
    admin_page = AdminPage(root)
    root.mainloop()
