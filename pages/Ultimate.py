import cv2, dlib, datetime, sqlite3, numpy as np, random, io, base64, pandas as pd, streamlit as st
from PIL import Image
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="centered")

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode()

# 업로드한 이미지 파일 경로
image_path = "bg/simple.jpg"  # 사용자가 업로드한 이미지 경로

# base64 인코딩된 이미지 가져오기
image_base64 = get_image_base64(image_path)

# CSS를 사용하여 배경 이미지 설정
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{image_base64}");
        background-size: cover;
        background-attachment: scroll;
        background-repeat: no-repeat;      
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# 데이터베이스 연결 및 테이블 생성
def init_db():
    conn = sqlite3.connect('personal_color.db')
    c = conn.cursor()
    
    # 테이블이 있는 경우 삭제
    # c.execute('DROP TABLE IF EXISTS color_results')
    
    c.execute('''CREATE TABLE IF NOT EXISTS color_results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  username TEXT,
                  age INTEGER,
                  gender TEXT,
                  avg_r REAL,
                  avg_g REAL,
                  avg_b REAL,
                  personal_color TEXT,
                  notes TEXT,
                  image BLOB)''')
    conn.commit()
    try:
        c.execute('ALTER TABLE color_results ADD COLUMN image BLOB')
        conn.commit()
    except sqlite3.OperationalError:
        pass
    return conn

# 결과 저장 함수
def save_result(conn, username, age, gender, avg_color, personal_color, notes, image):
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_data = image_to_binary(image)
    c.execute("INSERT INTO color_results (timestamp, username, age, gender, avg_r, avg_g, avg_b, personal_color, notes, image) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (timestamp, username, age, gender, avg_color[2], avg_color[1], avg_color[0], personal_color, notes, image_data))
    conn.commit()

def image_to_binary(image):
    buffer = io.BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    return buffer.getvalue()

# 데이터베이스에서 결과 조회
def get_results(conn):
    c = conn.cursor()
    c.execute("SELECT * FROM color_results ORDER BY timestamp DESC LIMIT 10")
    return c.fetchall()

def binary_to_image(binary):
    buffer = io.BytesIO(binary)
    return Image.open(buffer)

# dlib의 얼굴 감지기와 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

def mask_face_parts(frame, landmarks):
    # 랜드마크 인덱스 정의
    jaw_indices = list(range(0, 17))
    eye_indices = list(range(36, 48))
    nose_indices = list(range(27, 36))
    mouth_indices = list(range(48, 68))
    eyebrow_indices = list(range(17, 27))
    parts_indices = [eye_indices, nose_indices, mouth_indices, eyebrow_indices]

    # 마스크 생성
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_inv = np.ones(frame.shape[:2], np.uint8) * 255

    # 각 부분을 폴리곤으로 마스킹
    for indices in parts_indices:
        points = [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        cv2.fillPoly(mask_inv, [points], 0)

    # 얼굴 윤곽선 부분을 선으로 연결하여 폴리곤 생성
    jaw_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in jaw_indices]
    jaw_points.append((landmarks.part(0).x, landmarks.part(0).y))
    jaw_points = np.array(jaw_points, dtype=np.int32)
    cv2.fillPoly(mask_inv, [jaw_points], 0)

    # 눈, 코, 입 추가 마스킹
    eye_fill_points = [landmarks.part(i) for i in range(36, 42)]
    eye_fill_points = np.array([(p.x, p.y) for p in eye_fill_points], dtype=np.int32)
    cv2.fillPoly(mask, [eye_fill_points], 255)

    nose_fill_points = [landmarks.part(i) for i in range(27, 32)]
    nose_fill_points = np.array([(p.x, p.y) for p in nose_fill_points], dtype=np.int32)
    cv2.fillPoly(mask, [nose_fill_points], 255)

    outer_lip_indices = list(range(48, 60))
    lip_fill_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in outer_lip_indices]
    lip_fill_points.append((landmarks.part(48).x, landmarks.part(48).y))
    lip_fill_points = np.array(lip_fill_points, dtype=np.int32)
    cv2.fillPoly(mask, [lip_fill_points], 255)

    return mask, mask_inv


def get_random_colors(frame, mask, mask_inv, num_colors=40):
    combined_mask = cv2.bitwise_or(mask, mask_inv)
    non_masked_coords = np.where(combined_mask == 0)
    
    if len(non_masked_coords[0]) > num_colors:
        random_indices = random.sample(range(len(non_masked_coords[0])), num_colors)
        random_coords = [(non_masked_coords[0][i], non_masked_coords[1][i]) for i in random_indices]
    else:
        random_coords = list(zip(non_masked_coords[0], non_masked_coords[1]))
    
    random_colors = [frame[y, x] for y, x in random_coords]
    return random_colors


def train_random_forest():
    data = pd.read_csv('model\\color_df.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # 열 이름을 소문자로 변경
    X.columns = X.columns.str.lower()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)
    
    return rf_model, scaler, X.shape[1]

def classify_personal_colors(colors_df, rf_model, scaler, n_features):
    if n_features > 3:
        additional_features = pd.DataFrame(np.zeros((colors_df.shape[0], n_features - 3)), 
                                           columns=[f'feature_{i}' for i in range(3, n_features)])
        colors_df = pd.concat([colors_df, additional_features], axis=1)
    
    # 열 이름을 소문자로 변경
    colors_df.columns = colors_df.columns.str.lower()
    
    colors_scaled = scaler.transform(colors_df)
    predictions = rf_model.predict(colors_scaled)
    return predictions.tolist()

# 모델 학습
rf_model, scaler, n_features = train_random_forest()

def analyze_skin(face_region):
    # HSV 색공간으로 변환
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    
    # 피부 톤 분석
    hue = np.mean(hsv[:,:,0])
    saturation = np.mean(hsv[:,:,1])
    value = np.mean(hsv[:,:,2])
    
    # 밝기 분석
    brightness = np.mean(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
    
    # 균일성 분석 (표준편차 사용)
    uniformity = np.std(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
    
    return hue, saturation, value, brightness, uniformity

def recommend_products(hue, saturation, value, brightness, uniformity, gender):
    recommendations = []
    
    if gender == "여성":
        # 여성용 제품 추천
        if saturation < 50:
            recommendations.append("여성용 보습 크림 (건조한 피부용)")
        
        if saturation > 100:
            recommendations.append("여성용 오일 프리 모이스처라이저 (지성 피부용)")
        
        if brightness < 100:
            recommendations.append("여성용 비타민 C 세럼 (피부 톤 개선용)")
        
        if brightness > 150:
            recommendations.append("여성용 자외선 차단제 (높은 SPF)")
        
        if 0 <= hue < 10 or 160 <= hue < 180:
            recommendations.append("여성용 그린 컬러 코렉터 (붉은 기 개선용)")
        elif 10 <= hue < 25:
            recommendations.append("여성용 퍼플 컬러 코렉터 (노란 기 개선용)")
        
        if brightness < 100 and uniformity > 30:
            recommendations.append("여성용 레티놀 세럼 (안티에이징용)")
            
        
              
    elif gender == "남성":
        # 남성용 제품 추천
        if saturation < 50:
            recommendations.append("남성용 보습 로션 (건조한 피부용)")
        
        if saturation > 100:
            recommendations.append("남성용 오일 컨트롤 토너 (지성 피부용)")
        
        if brightness < 100:
            recommendations.append("남성용 브라이트닝 에센스 (피부 톤 개선용)")
        
        if brightness > 150:
            recommendations.append("남성용 선크림 (높은 SPF)")
        
        if brightness < 100 and uniformity > 30:
            recommendations.append("남성용 안티에이징 크림")
    
    return recommendations

def process_image(image, gender):
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        mask, mask_inv = mask_face_parts(frame, landmarks)
        random_colors = get_random_colors(frame, mask, mask_inv)

        personal_colors = classify_personal_colors(pd.DataFrame([color[::-1] for color in random_colors], columns=['r', 'g', 'b']), rf_model, scaler, n_features)
        color_counts = Counter(personal_colors)
        most_common_color = color_counts.most_common(1)[0][0]

        masked_frame = frame.copy()
        combined_mask = cv2.bitwise_or(mask, mask_inv)
        masked_frame[combined_mask == 255] = [0, 0, 0]

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(masked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 얼굴 영역 추출
        face_region = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(combined_mask))

        # 피부 분석
        hue, saturation, value, brightness, uniformity = analyze_skin(face_region)

        # 제품 추천
        recommendations = recommend_products(hue, saturation, value, brightness, uniformity, gender)

        masked_frame_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(masked_frame_rgb)

        original_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_image = Image.fromarray(original_frame_rgb)
        
        avg_color = np.mean(random_colors, axis=0)
        face_region = cv2.bitwise_and(frame, frame, mask=mask)
        hue, saturation, value, brightness, uniformity = analyze_skin(face_region)
        recommendations = recommend_products(hue, saturation, value, brightness, uniformity, gender)
        
        return image, result_image, avg_color, most_common_color, (hue, saturation, value, brightness, uniformity), recommendations
    return None, None, None, None, None, None

def capture_image():
    img_file_buffer = st.camera_input("사진 찍기")
    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        return img
    return None

def upload_image():
    uploaded_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        return img
    return None

def main():
    st.title("퍼스널 컬러 및 피부 분석")
    
    # personal_color 변수를 함수 시작 부분에서 초기화
    personal_color = None
    
    conn = init_db()

    username = st.text_input("이름")
    age = st.number_input("나이", min_value=0, max_value=120)
    gender = st.selectbox("성별", ["남성", "여성", "기타"])
    notes = st.text_area("추가 메모")

    col1, col2 = st.columns([5, 1])

    with col1:
        image_container = st.empty()

    option = st.radio("이미지 입력 방법 선택", ("이미지 업로드", "카메라로 찍기"))

    if option == "카메라로 찍기":
        captured_image = capture_image()
        if captured_image:
            original_image, result_image, avg_color, personal_color, skin_analysis, recommendations = process_image(captured_image, gender)
            if result_image:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_image, caption="원본 이미지")
                with col2:
                    st.image(result_image, caption="마스크 처리된 이미지")

                st.success("이미지가 성공적으로 처리되었습니다!")
                st.write(f"평균 RGB: {avg_color}")
                st.write(f"퍼스널 컬러: {personal_color}")
                
                # 피부 분석 결과와 이미지를 나란히 표시
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.subheader("피부 분석 결과:")
                    st.write(f"색조(Hue): {skin_analysis[0]:.2f}")
                    st.write(f"채도(Saturation): {skin_analysis[1]:.2f}")
                    st.write(f"명도(Value): {skin_analysis[2]:.2f}")
                    st.write(f"밝기: {skin_analysis[3]:.2f}")
                    st.write(f"균일성: {skin_analysis[4]:.2f}")
                with col2:
                # 퍼스널 컬러에 따라 다른 이미지 표시
                    if personal_color.lower() == 'spring':
                        st.image("season/sprWarm.png", caption="봄웜톤 컬러 팔레트")
                    elif personal_color.lower() == 'summer':
                        st.image("season/sumCool.png", caption="여름쿨톤 컬러 팔레트")
                    elif personal_color.lower() == 'autumn':
                        st.image("season/autWarm.png", caption="가을웜톤 컬러 팔레트")
                    elif personal_color.lower() == 'winter':
                        st.image("season/winCool.png", caption="겨울쿨톤 컬러 팔레트")
                
                st.subheader("추천 제품:")
                for product in recommendations:
                    st.write(f"- {product}")
                
                save_result(conn, username, age, gender, avg_color, personal_color, notes, captured_image)
            else:
                st.error("얼굴을 감지할 수 없습니다. 다시 시도해주세요.")
                
    elif option == "이미지 업로드":
        uploaded_image = upload_image()
        if uploaded_image:
            original_image, result_image, avg_color, personal_color, skin_analysis, recommendations = process_image(uploaded_image, gender)
            if result_image:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_image, caption="원본 이미지")
                with col2:
                    st.image(result_image, caption="마스크 처리된 이미지")

                st.success("이미지가 성공적으로 처리되었습니다!")
                st.write(f"평균 RGB: {avg_color}")
                st.write(f"퍼스널 컬러: {personal_color}")
                
                # 피부 분석 결과와 이미지를 나란히 표시
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.subheader("피부 분석 결과:")
                    st.write(f"색조(Hue): {skin_analysis[0]:.2f}")
                    st.write(f"채도(Saturation): {skin_analysis[1]:.2f}")
                    st.write(f"명도(Value): {skin_analysis[2]:.2f}")
                    st.write(f"밝기: {skin_analysis[3]:.2f}")
                    st.write(f"균일성: {skin_analysis[4]:.2f}")
                    
                with col2:
                # 퍼스널 컬러에 따라 다른 이미지 표시
                    if personal_color.lower() == 'spring':
                        st.image("season/sprWarm.png", caption="봄웜톤 컬러 팔레트")
                    elif personal_color.lower() == 'summer':
                        st.image("season/sumCool.png", caption="여름쿨톤 컬러 팔레트")
                    elif personal_color.lower() == 'autumn':
                        st.image("season/autWarm.png", caption="가을웜톤 컬러 팔레트")
                    elif personal_color.lower() == 'winter':
                        st.image("season/winCool.png", caption="겨울쿨톤 컬러 팔레트")
                        
                st.subheader(f"당신의 퍼스널 컬러는 {personal_color}입니다. 이에 맞는 추천 제품은 {personal_color} 에서 참고해보세요!")        
                        
                st.subheader("추천 제품:")
                for product in recommendations:
                    st.write(f"- {product}")
                
                save_result(conn, username, age, gender, avg_color, personal_color, notes, uploaded_image)
            else:
                st.error("얼굴을 감지할 수 없습니다. 다시 시도해주세요.")

    st.subheader("최근 분석 결과")
    results = get_results(conn)
    for result in results:
        st.write(f"시간: {result[1]}, 이름: {result[2]}, 나이: {result[3]}, 성별: {result[4]}, 퍼스널 컬러: {result[8]}, 메모: {result[9]}")
        if result[10]:
            image = binary_to_image(result[10])
            st.image(image)

    conn.close()

if __name__ == "__main__":
    main()
