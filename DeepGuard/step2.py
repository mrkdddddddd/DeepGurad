import cv2
import numpy as np
import mediapipe as mp
import os

# 증폭
# 이미지를 로드하고 디렉토리에서 파일명 리스트를 가져옴
photo_dir = "./faces/"
changed_photo_dir = "./changed_photos/"
photo_names = [f for f in os.listdir(photo_dir) if f.endswith('.jpg')]

if not os.path.exists("changed_photos"):
    os.makedirs("changed_photos")

# 이미지 증폭을 위한 함수 정의
def augment_image(image):
    augmented_images = []
    
    # 원본 이미지 추가
    augmented_images.append(image)
    
    # 이미지 반전
    flipped_image = cv2.flip(image, 1)
    augmented_images.append(flipped_image)
    
    # 이미지 회전 및 기울기 변화
    rows, cols, _ = image.shape
    for angle in range(-10, 11, 5):
        M_rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, M_rotation, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        augmented_images.append(rotated_image)
    
    # # 줌인, 줌아웃
    # for scale_factor in [0.9, 1.1]:
    #     scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    #     augmented_images.append(scaled_image)
    
    # shear 조절
    shear_range = 10
    for shear in range(-shear_range, shear_range + 1, 5):
        M_shear = np.float32([[1, shear / 100, 0], [0, 1, 0]])
        sheared_image = cv2.warpAffine(image, M_shear, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        augmented_images.append(sheared_image)
    
    # 밝기 조절
    for brightness in [-50, 50]:
        brightened_image = cv2.convertScaleAbs(image, beta=brightness)
        augmented_images.append(brightened_image)
    
    return augmented_images

# 사진 증폭 작업 수행
for photo_name in photo_names:
    # 이미지 로드
    photo_path = os.path.join(photo_dir, photo_name)
    image = cv2.imread(photo_path)
    
    # 이미지 증폭
    augmented_images = augment_image(image)
    
    # 증폭된 이미지 저장
    base_name, ext = os.path.splitext(photo_name)
    for idx, augmented_image in enumerate(augmented_images):
        augmented_photo_name = f"{base_name}_augmented_{idx}{ext}"
        augmented_photo_path = os.path.join(changed_photo_dir, augmented_photo_name)
        cv2.imwrite(augmented_photo_path, augmented_image)

# # 랜드마크
# # Mediapipe의 얼굴 랜드마크 모듈 로드
# mp_face_mesh = mp.solutions.face_mesh

# # 랜드마크 추출을 위한 객체 생성
# face_mesh = mp_face_mesh.FaceMesh()

# # 이미지가 저장된 폴더 경로
# folder_path = './changed_photos/'

# # 폴더 내의 모든 이미지 파일 이름 리스트 가져오기
# image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# # 이미지 파일들을 순회하며 얼굴 랜드마크 추출
# for image_file in image_files:
#     image_path = os.path.join(folder_path, image_file)
    
#     # 이미지 읽기
#     image = cv2.imread(image_path)
    
#     # 이미지를 RGB로 변환
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # 얼굴 랜드마크 추출
#     results = face_mesh.process(image_rgb)
    
#     # 랜드마크 좌표 추출 및 표시
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             for landmark in face_landmarks.landmark:
#                 x = landmark.x
#                 y = landmark.y
                
#                 # 랜드마크 좌표 표시 (이미지 위에)
#                 cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 1, (0, 255, 0), -1)
    
#     # 결과 이미지 저장
#     if not os.path.exists("output"):
#         os.makedirs("output")
#     output_path = os.path.join('output/', image_file)
#     cv2.imwrite(output_path, image)