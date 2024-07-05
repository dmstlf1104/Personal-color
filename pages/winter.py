import os
import streamlit as st
from PIL import Image

# Define the directory and file names
directory = 'Winter'
file_names = ["28.jpeg", "29.jpeg", "30.jpeg", "31.jpeg", "32.jpeg", "33.jpeg", "34.jpeg", "35.jpeg", "36.jpeg"]
cosmetic_names = ["어뮤즈 세라믹 스킨 퍼펙터<br> 쿠션 1.5호", "퓌 멜로우 듀얼 블러셔<br> 아이시 큐피트", 
                  "클리오 킬브로우 쉐이핑<br> 파우더 브로우", "롬앤 쥬시래스팅 틴트<br> 25 베어그레이프",
                  "롬앤 아이섀도우 배러 댄<br> 팔레트 07 베리 푸시아 가든", 
                  "클리오 프리즘 하이라이터<br> 듀오 02 라벤더 보야지", "슈에무라 블러셔 M225 <br>라벤더 할로",
                    "에뛰드 픽싱 틴트 바<br> 라이블리 제드", "롬앤 블러 퍼지 틴트 11<br>푸시아 바이브"]

# Define the size to resize images to
resize_size = (300, 300)

def display_images_with_titles(directory, file_names, cosmetic_names, size):
    rows = [st.columns([1, 30, 1, 30, 1]) for _ in range(3)]  # 3x3 그리드 생성, 간격을 위해 더 많은 열 생성
    col_positions = [0, 2, 4]  # 이미지가 위치할 열의 인덱스

    for idx, (file_name, cosmetic_name) in enumerate(zip(file_names, cosmetic_names)):
        row = idx // 3
        col = col_positions[idx % 3]
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            image = Image.open(file_path).resize(size)
            with rows[row][col]:
                st.image(image, use_column_width=False)
                st.markdown(
                    f"<div style='text-align: center; font-size: 24px; white-space: nowrap;'>{cosmetic_name}</div>", 
                    unsafe_allow_html=True
                )

def main():
    st.set_page_config(layout="centered")
    
    st.title("Autumn 퍼스널 컬러 화장품 추천")
    
    # Display images with titles
    display_images_with_titles(directory, file_names, cosmetic_names, resize_size)


    st.write("Winter 타입의 메이크업 팁:")
    st.write("1. 선명하고 쿨한 색조를 선택하세요.")
    st.write("2. 실버 톤의 하이라이터를 사용하세요.")
    st.write("3. 딥 레드나 핑크 계열의 블러셔가 잘 어울립니다.")
    
if __name__ == "__main__":
    main()
