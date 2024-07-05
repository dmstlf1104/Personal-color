import streamlit as st
from PIL import Image


st.set_page_config(page_title="Summer, Winter", page_icon="2")

# 상단에 두 개의 열을 만듭니다
header_left, header_right = st.columns([8, 1])

# 오른쪽 열에 버튼을 배치합니다
with header_right:
    if st.button("Main"):
        st.switch_page("main.py")

# 메인 콘텐츠 영역
st.title("Summer, Winter")

# 이미지 파일 경로
image_path = "image_inv/color2.png"  # 실제 이미지 파일 경로로 변경해주세요

# 이미지 표시
try:
    image = Image.open(image_path)
    st.image(image, caption="퍼스널 컬러란?", width=500)
except FileNotFoundError:
    st.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
except Exception as e:
    st.error(f"이미지를 불러오는 중 오류가 발생했습니다: {e}")

# 이미지 아래에 설명 추가
st.write("""
         
- ### 여름 : 블루베이스, 서브는 화이트!
★ **해가 가장김 ☞ 빛이 쎄게 들어와서 눈이부셔 눈앞이 뿌옇게 보임 ☞ 쿨톤베이스의 고명도  추천!!** ★

- ### 겨울 : 블루베이스 서브는 블랙!
★ **눈의 반짝거리는 느낌을 표현 ☞ 쿨톤베이스의 고채도  추천!!** ★

\n ## 자세한 내용은 아래 링크!! \n
https://cafe.naver.com/b00k2012/568113?art=ZXh0ZXJuYWwtc2VydmljZS1uYXZlci1zZWFyY2gtY2FmZS1wcg.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjYWZlVHlwZSI6IkNBRkVfVVJMIiwiY2FmZVVybCI6ImIwMGsyMDEyIiwiYXJ0aWNsZUlkIjo1NjgxMTMsImlzc3VlZEF0IjoxNzE5OTAyNzc0ODQzfQ.vGv0S0_o2F1-j9mhRs4ilD3OoMth7Wtpu-8_WzJ-LUI
""")

# 추가적인 정보나 기능을 넣고 싶다면 이 아래에 계속 작성할 수 있습니다.