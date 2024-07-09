import streamlit as st, time, os, base64, numpy as np
from PIL import Image


st.set_page_config(layout="centered")

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode()

# 업로드한 이미지 파일 경로
image_path = "bg/green.jpg"  # 사용자가 업로드한 이미지 경로

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

# 중앙 정렬을 위한 컨테이너
st.markdown('<div class="main-content">', unsafe_allow_html=True)

def resize_and_pad(img, target_size, background_color=(255, 255, 255)):  # 배경색은 흰색으로 설정
    width, height = img.size
    ratio = min(target_size[0] / width, target_size[1] / height)
    new_size = (int(width * ratio), int(height * ratio))
    img = img.resize(new_size, Image.LANCZOS)
    
    new_img = Image.new("RGB", target_size, background_color)
    position = ((target_size[0] - new_size[0]) // 2,
                (target_size[1] - new_size[1]) // 2)
    new_img.paste(img, position)
    return new_img

def load_images(image_directory, target_size=(1000,1000)):
    images = []
    for filename in os.listdir(image_directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(image_directory, filename)
            img = Image.open(img_path)
            img = resize_and_pad(img, target_size)
            name_without_extension = os.path.splitext(filename)[0]
            images.append((name_without_extension, img))
    return images

def main():
    st.title("당신의 퍼스널컬러는? ")

    # 이미지가 저장된 디렉토리 경로
    image_directory = "image_directory"  # 실제 이미지 폴더 경로로 변경하세요
    target_size=(1000,1000)
    images = load_images(image_directory, target_size)

    if not images:
        st.write("이미지를 찾을 수 없습니다.")
        return

    # 사이드바에 컨트롤 추가
    with st.sidebar:
        slide_speed = st.slider("슬라이드 속도 (초)", min_value=1, max_value=10, value=3)
        auto_play = st.checkbox("자동 재생", value=True)

    # 메인 영역을 두 열로 나눔
    col1, col2 = st.columns([5, 1])


    # 이미지를 표시할 빈 공간 생성
    with col1:
        image_container = st.empty()
        caption_container = st.empty()

    # 현재 이미지 인덱스
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    while True:
        # 현재 이미지와 캡션 표시
        with col1:
            image_container.image(images[st.session_state.current_index][1])
            # caption_container.write(f"이미지 {st.session_state.current_index + 1}/{len(images)}")

        # 자동 재생 모드
        if auto_play:
            time.sleep(slide_speed)
            st.session_state.current_index = (st.session_state.current_index + 1) % len(images)
        # else:
        #     # 수동 모드
        #     st.session_state.current_index = st.slider("이미지 선택", 0, len(images) - 1, st.session_state.current_index)
        #     time.sleep(0.1)  # 슬라이더 반응성 개선

        # Streamlit의 재실행을 방지하기 위한 빈 요소
        st.empty()

if __name__ == "__main__":
    main()