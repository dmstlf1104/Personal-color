import os
import streamlit as st
from PIL import Image

# Define the directory and file names
directory = 'Summer'
file_names = ["10.png", "11.jpeg", "12.jpeg", "13.jpeg", "14.jpeg", "15.jpeg", "16.png", "17.png", "18.png"]
cosmetic_names = ["롬앤 제로 매트 립스틱 01.<br>더스티 핑크", "맥 703 런웨이 히트", 
                  "에뛰드 그림자 쉐딩 3호 재조명", "에스쁘아 프로 테일러 <br>비 글로우 쿠션 아이보리", 
                  "에스티로더 더블 웨어 <br>파운데이션 쿨 바닐라", "클리오 프로 아이 팔레트 <br>14한남동 아뜰리에", 
                  "태그 무드 블러쉬 빔 <br>2호 페어 모브","페리페라 20 여주등극", "페리페라 21 생기대란"]

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
    
    st.title("Summer 퍼스널 컬러 화장품 추천")
    
    # Display images with titles
    display_images_with_titles(directory, file_names, cosmetic_names, resize_size)


    st.write("Summer 타입의 메이크업 팁:")
    st.write("1. 시원하고 부드러운 파스텔 컬러를 선택하세요.")
    st.write("2. 실버 톤의 하이라이터를 사용하세요.")
    st.write("3. 라벤더나 핑크 계열의 블러셔가 잘 어울립니다.")

    
if __name__ == "__main__":
    main()
