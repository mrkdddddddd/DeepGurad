import mediapipe as mp
import cv2
import os

# Mediapipe의 얼굴 랜드마크 모듈 로드
mp_face_mesh = mp.solutions.face_mesh

# 랜드마크 추출을 위한 객체 생성
face_mesh = mp_face_mesh.FaceMesh()

# 이미지가 저장된 폴더 경로
folder_path = './changed_photos/'

# 폴더 내의 모든 이미지 파일 이름 리스트 가져오기
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 이미지 파일들을 순회하며 얼굴 랜드마크 추출
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    
    # 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 얼굴 랜드마크 추출
    results = face_mesh.process(image_rgb)
    
    # 랜드마크 좌표 추출 및 표시
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                
                # 랜드마크 좌표 표시 (이미지 위에)
                cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 1, (0, 255, 0), -1)
    
    # 결과 이미지 저장
    if not os.path.exists("output"):
        os.makedirs("output")
    output_path = os.path.join('output/', image_file)
    cv2.imwrite(output_path, image)