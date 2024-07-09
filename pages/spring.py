import os
import streamlit as st
from PIL import Image

# Define the directory and file names
directory = 'Warm'
file_names = ["1.jpeg", "2.jpeg", "3.jpg", "4.jpeg", "5.jpeg", "6.jpg", "7.jpeg", "8.png", "9.jpg"]
cosmetic_names = ["시크릿 스킨 메이커 제로<br>리퀴드 21호", "아리따움 젤리바 <br>오렌지 칼테일", 
                  "아트 클래스 바이<br> 로뎅 쉐이딩", "이프노즈 돌 아이 마스카라", "크리니크 치크 멜론 팝", 
                  "클라란스 인스턴트 <br>립 오일 레드베리 ", "포켓 팔레트 블루밍",
                    "플린 다이브 워터 틴트 3.2g", "히로인 스무스 리퀴드 <br>아이라이너 브라운"]

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
    
    st.title("Spring 퍼스널 컬러 화장품 추천")
    
    # Display images with titles
    display_images_with_titles(directory, file_names, cosmetic_names, resize_size)


    st.write("Spring 타입의 메이크업 팁:")
    st.write("1. 밝고 화사한 컬러를 선택하세요.")
    st.write("2. 골드 톤의 하이라이터를 사용하세요.")
    st.write("3. 피치나 코랄 계열의 블러셔가 잘 어울립니다.")
    
if __name__ == "__main__":
    main()
