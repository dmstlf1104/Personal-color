import streamlit as st
from PIL import Image
import base64

st.set_page_config(page_title="Spring, Authmn", page_icon="🌸")

# 상단에 두 개의 열을 만듭니다
header_left, header_right = st.columns([8, 1])

# 오른쪽 열에 버튼을 배치합니다
with header_right:
    if st.button("Main"):
        st.switch_page("main.py")

# 메인 콘텐츠 영역
st.title("Spring , Authmn")

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

# 중앙 정렬을 위한 컨테이너
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# 이미지 파일 경로
image_path = "image_inv/color1.png"  # 실제 이미지 파일 경로로 변경해주세요

# 이미지 표시
try:
    image = Image.open(image_path)
    st.image(image, caption="퍼스널 컬러란?", width=700)
except FileNotFoundError:
    st.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
except Exception as e:
    st.error(f"이미지를 불러오는 중 오류가 발생했습니다: {e}")

# 이미지 아래에 설명 추가
st.write("""
- ### 봄 : 옐로우베이스 
★ **겨울에서 봄이 되며 해가 길어짐 ☞ 빛이 들어옴 ☞ 웜톤베이스의 고명도  추천!!** ★

- ### 가을 : 옐로우, 골드베이스
★ **겨울보단 짧지만 어둠이 길어지기 시작 ☞ 다크그레이 섞인 저명도 추천!!** ★

\n ## 자세한 내용은 아래 링크!! \n
[링크 클릭하기](https://cafe.naver.com/b00k2012/568113?art=ZXh0ZXJuYWwtc2VydmljZS1uYXZlci1zZWFyY2gtY2FmZS1wcg.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjYWZlVHlwZSI6IkNBRkVfVVJMIiwiY2FmZVVybCI6ImIwMGsyMDEyIiwiYXJ0aWNsZUlkIjo1NjgxMTMsImlzc3VlZEF0IjoxNzE5OTAyNzc0ODQzfQ.vGv0S0_o2F1-j9mhRs4ilD3OoMth7Wtpu-8_WzJ-LUI)
""")

# 중앙 정렬을 위한 컨테이너 닫기
st.markdown('</div>', unsafe_allow_html=True)
