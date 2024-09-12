import streamlit as st
import requests
from PIL import Image
import io

# FastAPI 서버 URL 설정
API_URL = "http://127.0.0.1:8000/predict-image"

# Streamlit 앱 설정
st.title("MNIST Image Prediction")
st.write("Upload an image of a digit, and the model will predict the digit.")

# 이미지 업로드 기능
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# 업로드된 파일이 있을 경우
if uploaded_file is not None:
    try:
        # PIL을 사용하여 이미지를 열고 화면에 표시
        cols = st.columns(2)
        with cols[0]:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        with cols[1]:
            # "Predict" 버튼
            if st.button("Predict"):
                # 이미지를 바이트로 변환
                image_bytes = io.BytesIO()
                image.save(image_bytes, format=image.format)
                image_bytes = image_bytes.getvalue()

                # FastAPI 서버에 요청 보내기
                response = requests.post(
                    API_URL,
                    files={"file": ("image", image_bytes, uploaded_file.type)}  # 파일 데이터를 "file"이라는 키로 전송
                )

                # 응답이 성공적일 때
                if response.status_code == 200:
                    result = response.json()
                    probabilities = result.get("probabilities", [])
                    predicted_class = result.get("predicted_class", -1)
                    
                    # 확률과 예측 클래스 레이블 출력
                    st.success(f"Predicted Class: {predicted_class}")
                    st.write("Class Probabilities:")
                    for i, prob in enumerate(probabilities):
                        st.write(f"Class {i}: {prob:.4f}")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
