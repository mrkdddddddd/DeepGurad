import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 웹캠에서 프레임 읽기
cap = cv2.VideoCapture(0)

# Mediapipe 얼굴 감지 모델 초기화
with mp_face_detection.FaceDetection(min_detection_confidence=1) as face_detection:

    # 승인 여부 변수 초기화
    # 테스트용 기본 False가 맞음
    accept_id = True
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from the camera.")
            break

        # 프레임을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 감지 수행
        results = face_detection.process(rgb_frame)

        # 얼굴이 감지된 경우
        if results.detections:
            cv2.waitKey(1000)  # 1초 대기
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                # 얼굴 주변에 네모 박스 그리기
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 로딩 중 메시지 표시
                cv2.waitKey(500)  # 0.5초 대기
                cv2.putText(frame, 'Loading...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Camera', frame)
                cv2.waitKey(2000)  # 2초 대기
                
                # 결과 메시지 표시
                if accept_id:
                    message = 'Success'
                else:
                    message = 'Fail'
                cv2.putText(frame, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Camera', frame)
                cv2.waitKey(2000)  # 2초 대기

                # 다시 처음으로 돌아가기 위해 continue 사용
                continue

        # 얼굴이 감지되지 않은 경우 프레임 표시
        cv2.imshow('Camera', frame)

        # 'esc' 키를 누르거나 창의 닫힘 버튼을 클릭하면 종료
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()