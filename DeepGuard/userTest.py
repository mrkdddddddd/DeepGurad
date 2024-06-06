import tkinter as tk
import cv2
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageTk

# 사용자용

models_path = 'models'
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

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        self.root.geometry("800x600")
        self.root.configure(bg='#404040')  # 배경색을 어두운 회색으로 설정

        self.video_label = tk.Label(root, bg='#404040')  # 비디오 라벨의 배경색도 어두운 회색으로 설정
        self.video_label.pack()

        # 나가기 버튼을 동그란 버튼으로 생성하고 배경색과 글자색을 설정
        self.exit_button = tk.Button(root, text="나가기", command=self.on_closing, bg='#404040', fg='white', width=10, height=2, relief="raised", borderwidth=0, highlightthickness=0, highlightbackground='white')
        self.exit_button.pack(pady=10)

        self.cap = cv2.VideoCapture(0)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
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

            except:
                cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
