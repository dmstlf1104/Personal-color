import os
import streamlit as st
from PIL import Image

# Define the directory and file names
directory = 'Autumn'
file_names = ["19.jpeg", "20.jpeg", "21.jpeg", "22.jpeg", "23.jpeg", "24.jpeg", "25.jpeg", "26.jpeg", "27.jpeg"]
cosmetic_names = ["맥 파우더 키스 벨벳 <br>블러 슬럼 스틱 멀잇오버", "헤라 팜파스", 
                  "입큰 퍼스널 무드 워터핏 <br>쉬어 립틴트 누디피칸", "롬앤 블러 퍼지 틴트 포멜로코", 
                  "크리니크 누드 팝","아임미미 아임 애프터눈티 <br>블러셔 팔레트 밀크티타임", 
                  "어바웃톤 플러피 웨어<br> 블러셔 베일 피치","에스쁘아 리얼 아이팔레트<br> 오트라떼", 
                  "데이지크 섀도우 팔레트 밀크라떼"]

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


    st.write("Autumn 타입의 메이크업 팁:")
    st.write("1. 따뜻하고 풍부한 색조를 선택하세요.")
    st.write("2. 브론즈 톤의 하이라이터를 사용하세요.")
    st.write("3. 테라코타나 오렌지 계열의 블러셔가 잘 어울립니다.")

    
if __name__ == "__main__":
    main()
