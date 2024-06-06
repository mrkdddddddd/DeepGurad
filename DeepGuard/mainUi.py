import tkinter as tk
from tkinter import *
from tkinter import Button, Label
import cv2
import os
from PIL import ImageTk, Image
import threading
import time
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def capture_images():
    # 사진을 저장할 디렉토리 생성
    if not os.path.exists("photos"):
        os.makedirs("photos")
    
    # 웹캠 연결
    cap = cv2.VideoCapture(0)

    # 사진 인덱스 초기화
    photo_index = 0

    # 카메라 화면 실시간 표시 및 사진 촬영
    while True:
        ret, frame = cap.read()

        if not ret:
            print("카메라 연결 실패")
            break

        # 화면에 프레임 표시
        cv2.imshow("Capture Camera", frame)

        # 화면 갱신
        cv2.waitKey(3)
        
        # Mediapipe 얼굴 감지 모델 초기화
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            
            # 프레임을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 얼굴 감지 수행
            results = face_detection.process(rgb_frame)

            # 얼굴이 감지된 경우
            if results.detections:
        
                # 사진 촬영 및 저장
                photo_path = f"photos/photo_{photo_index}.jpg"
                cv2.imwrite(photo_path, frame)
                print(f"Captured photo: {photo_path}")
                photo_index += 1
                
                # 찍는 중 메시지 표시
                print("찍는중...")

                # 팝업 창 생성
                popup = Toplevel()
                popup.title("Capturing Photos")
                popup.geometry("300x100")
                
                # 찍는 중 메시지 표시
                Label(popup, text="찍는 중", font=("Helvetica", 12)).pack(pady=10)
                
                # 2초 대기 후 팝업 창 닫기
                popup.after(2000, popup.destroy)
                
                # 2초 대기 후 다음 사진 촬영
                time.sleep(2)
            else:
                if cv2.waitKey(1) == 27:
                    break
                continue
        
        # 모든 사진을 찍은 경우
        if photo_index >= 1:
            # 완료 메시지 표시
            print("완료되었습니다.")

            # 팝업 창 생성
            popup2 = Toplevel()
            popup2.title("Finish Capturing Photos")
            popup2.geometry("300x100")
            
            # 완료되었습니다 메시지 표시
            Label(popup2, text="완료되었습니다.", font=("Helvetica", 12)).pack(pady=10)
            
            # 'esc' 키를 누르거나 창의 닫힘 버튼을 클릭하면 종료
            if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Finish Capturing Photos', cv2.WND_PROP_VISIBLE) < 1:
                break
            else:
                # 2초 대기 후 팝업 창 닫기
                popup2.after(2000, popup2.destroy)
                break
        continue

    # 카메라 연결 해제
    cap.release()
    
    # 창 닫기
    cv2.destroyAllWindows()

def start_capture():
    capture_thread = threading.Thread(target=capture_images)
    capture_thread.start()

win = tk.Tk()
win.title("DeepGuard")
win.geometry("645x540+30+30")
win.resizable(False, False)

lbl = tk.Label(win, text="얼굴 인식 Live")
lbl.grid(row=0, column=0)

# 'Start Video' 버튼 추가
btn_start = tk.Button(win, text="Capture Video", command=start_capture)
btn_start.grid(row=1, column=0)

frm = tk.Frame(win, bg="white", width=400, height=440)
frm.grid(row=2, column=0)

lbl1 = tk.Label(frm)
lbl1.grid()

cap = cv2.VideoCapture(0)

def video_play():
    global cap  # cap 변수를 global로 선언
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cap = cv2.VideoCapture(0)
        # return
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl1.imgtk = imgtk
    lbl1.configure(image=imgtk)
    lbl1.after(10, video_play)

video_play()

win.mainloop()