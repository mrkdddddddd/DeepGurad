import cv2
import os
import time
import threading
from tkinter import *
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 사진을 저장할 디렉토리 생성
if not os.path.exists("photos"):
    os.makedirs("photos")

# 카메라 캡처 함수
def capture_images():
    # 웹캠 연결
    cap = cv2.VideoCapture(0)
    
    # 카메라 화면 표시 창 생성
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 800, 600)
    
    # 사진 인덱스 초기화
    photo_index = 0
    
    # 시작 버튼 비활성화
    start_button.config(state=DISABLED)
    
    # 5초 대기 후 사진 촬영 시작
    time.sleep(5)
    
    # 카메라 화면 실시간 표시 및 사진 촬영
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame from the camera.")
            break
        
        # 화면에 프레임 표시
        cv2.imshow("Camera", frame)
        
        # 화면 갱신
        cv2.waitKey(1)

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
                continue
        
        # 모든 사진을 찍은 경우
        if photo_index >= 10:
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
                popup2.after(8000, popup2.destroy)
                break

        continue
    
    # 카메라 연결 해제
    cap.release()

    # 창 닫기
    cv2.destroyAllWindows()


# 버튼 클릭 시 사진 촬영 시작
def start_capture():
    capture_thread = threading.Thread(target=capture_images)
    capture_thread.start()

# 메인 윈도우 생성
root = Tk()
root.title("Capture Photos")

# 시작 버튼 생성
start_button = Button(root, text="Start", command=start_capture)
start_button.pack(pady=20)

# 메인 윈도우 표시
root.mainloop()
