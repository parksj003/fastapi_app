from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import torch
from torch import nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import io

# SimpleCNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 채널 수정 (1채널 입력)
        self.fc = nn.Linear(16 * 64 * 64, 10)  # 이미지 크기 (64x64)와 출력 클래스 수 (10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# MNIST 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 모델 입력 크기에 맞게 리사이즈
    transforms.ToTensor(),        # 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

# 모델 초기화 및 가중치 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))  # 디바이스에 맞게 로드
model.eval()  # 평가 모드

# FastAPI 애플리케이션 생성
app = FastAPI()

# 비동기 모델 예측 함수
async def predict_image(image: Image.Image) -> dict:
    try:
        # 이미지를 'L' 모드(그레이스케일)로 변환
        image = image.convert("L")
        # PyTorch 전처리 적용
        image_tensor = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가
        
        # 모델 추론
        with torch.no_grad():
            output = model(image_tensor)
        
        # 소프트맥스 적용하여 확률로 변환
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()
        predicted_class = probabilities.index(max(probabilities))
        
        return {"probabilities": probabilities, "predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

# 예측 엔드포인트 정의
@app.post("/predict-image")
async def upload_image(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        # 이미지 파일 열기
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.convert("L")  # 이미지를 'L' 모드(그레이스케일)로 변환
    except (UnidentifiedImageError, IOError):
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

    # 모델 추론 호출
    prediction = await predict_image(image)
    return prediction

# 서버 실행 (테스트 시 필요하면 사용)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("be_main:app", host="127.0.0.1", port=8000, reload=True)
