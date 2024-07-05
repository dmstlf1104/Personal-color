import cv2
import dlib

# dlib의 얼굴 감지기와 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model\\shape_predictor_68_face_landmarks.dat")

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = detector(gray)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 랜드마크 감지
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    # 결과 영상 표시
    cv2.imshow("Face Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 및 모든 창 종료
cap.release()
cv2.destroyAllWindows()
