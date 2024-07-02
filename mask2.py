import cv2
import dlib
import numpy as np

# dlib의 얼굴 감지기와 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model\\shape_predictor_68_face_landmarks.dat")

# 이미지 파일 로드
image_path = 'image\\ga.jpg'
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from path: {image_path}")
    exit()

def mask_face_parts(frame, landmarks):
    # 랜드마크 인덱스 정의
    jaw_indices = list(range(0, 17))  # 얼굴 윤곽선 (턱선)
    eye_indices = list(range(36, 48))  # 눈
    nose_indices = list(range(27, 36))  # 코
    mouth_indices = list(range(48, 68))  # 입
    eyebrow_indices = list(range(17, 27))  # 눈썹
    parts_indices = [eye_indices, nose_indices, mouth_indices, eyebrow_indices]

    # 마스크 생성
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_inv = np.ones(frame.shape[:2], dtype=np.uint8) * 255

    # 각 부분을 폴리곤으로 마스킹
    for indices in parts_indices:
        points = [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        cv2.fillPoly(mask_inv, [points], 0)

    # 얼굴 윤곽선 부분을 선으로 연결하여 폴리곤 생성
    jaw_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in jaw_indices]
    jaw_points.append((landmarks.part(0).x, landmarks.part(0).y))  # 0번과 16번을 연결하여 폴리곤을 닫음
    jaw_points = np.array(jaw_points, dtype=np.int32)
    cv2.fillPoly(mask_inv, [jaw_points], 0)  # 윤곽선 외부를 검은색으로 마스킹

    # 추가 마스킹 (36번과 41번을 이어서 마스킹)
    eye_fill_points = [
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(37).x, landmarks.part(37).y),
        (landmarks.part(38).x, landmarks.part(38).y),
        (landmarks.part(39).x, landmarks.part(39).y),
        (landmarks.part(40).x, landmarks.part(40).y),
        (landmarks.part(41).x, landmarks.part(41).y)
    ]
    eye_fill_points = np.array(eye_fill_points, dtype=np.int32)
    cv2.fillPoly(mask, [eye_fill_points], 255)

    # 추가 마스킹 (27번과 31번을 이어서 마스킹)
    nose_fill_points = [
        (landmarks.part(27).x, landmarks.part(27).y),
        (landmarks.part(28).x, landmarks.part(28).y),
        (landmarks.part(29).x, landmarks.part(29).y),
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(31).x, landmarks.part(31).y)
    ]
    nose_fill_points = np.array(nose_fill_points, dtype=np.int32)
    cv2.fillPoly(mask, [nose_fill_points], 255)

    # 입술 마스킹 (48번부터 59번을 이은 후 48번과 59번을 연결하여 마스킹)
    outer_lip_indices = list(range(48, 60))
    lip_fill_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in outer_lip_indices]
    lip_fill_points.append((landmarks.part(48).x, landmarks.part(48).y))  # 48번과 59번을 연결
    lip_fill_points = np.array(lip_fill_points, dtype=np.int32)
    cv2.fillPoly(mask, [lip_fill_points], 255)

    return mask, mask_inv

def get_average_rgb(frame, mask, mask_inv):
    # 마스크가 적용되지 않은 부분의 RGB 값 추출
    combined_mask = cv2.bitwise_or(mask, mask_inv)
    non_masked_pixels = frame[combined_mask == 0]
    if non_masked_pixels.size > 0:
        avg_color = np.mean(non_masked_pixels, axis=0)
    else:
        avg_color = [0, 0, 0]
    return avg_color

# 단일 이미지 파일을 처리하는 루프
while True:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = detector(gray)
    for face in faces:
        # 얼굴 랜드마크 감지
        landmarks = predictor(gray, face)
        
        # 얼굴 특정 부분 마스킹 및 랜드마크 번호 표시
        mask, mask_inv = mask_face_parts(frame, landmarks)

        # 마스크가 적용되지 않은 부분의 평균 RGB 값 추출
        avg_color = get_average_rgb(frame, mask, mask_inv)
        print(f"Average RGB: {avg_color}")

        # 원본 이미지에 마스크 적용
        masked_frame = frame.copy()
        combined_mask = cv2.bitwise_or(mask, mask_inv)
        masked_frame[combined_mask == 255] = [0, 0, 0]

        # 얼굴 영역에 사각형 그리기
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(masked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 결과 영상 표시
    cv2.imshow("Masked Face Parts", masked_frame)

    # 'q' 키를 눌러서 종료
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# 모든 창 종료
cv2.destroyAllWindows()
