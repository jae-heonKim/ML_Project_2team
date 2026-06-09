# FastAPI 기반 한국어 이메일 긴급도 분류 API 서버
# n8n에서 HTTP Request로 호출하기 위한 모델 서버 코드

# FastAPI 웹 서버 라이브러리
from fastapi import FastAPI

# 요청 데이터 형식 정의용
from pydantic import BaseModel

# PyTorch
import torch

# HuggingFace Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# 학습 완료된 모델 폴더 경로
MODEL_PATH = "C:\기계학습 개인보고서/final_model"

# FastAPI 앱 생성
app = FastAPI()

# GPU 사용 가능 여부 확인
# CUDA가 없으면 CPU 사용
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 저장된 tokenizer 불러오기
# tokenizer는 문장을 token id로 변환하는 역할
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

# 저장된 BERT 분류 모델 불러오기
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# 모델을 GPU 또는 CPU로 이동
model.to(device)

# 평가 모드 활성화
# dropout 등의 학습 기능 비활성화
model.eval()


# HTTP 요청 데이터 형식 정의
# n8n이 아래 형태로 요청을 보내게 됨
#
# {
#   "text": "오늘 중으로 회신 부탁드립니다."
# }

class EmailRequest(BaseModel):

    # 이메일 텍스트
    text: str


# POST 방식 API 생성
# 주소:
# http://localhost:8000/predict

@app.post("/predict")
def predict_email(request: EmailRequest):

    # 요청에서 이메일 텍스트 추출
    text = request.text

    # 입력 문장 토큰화
    # BERT가 이해할 수 있는 숫자 형태로 변환
    inputs = tokenizer(

        text,

        # PyTorch tensor 형식 반환
        return_tensors="pt",

        # 최대 길이 초과 시 자르기
        truncation=True,

        # padding 적용
        padding=True,

        # 최대 토큰 길이
        max_length=256,
    )

    # GPU / CPU 이동
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    # gradient 계산 비활성화
    # 추론(inference) 단계에서는 학습이 필요 없음
    with torch.no_grad():

        # 모델 예측 수행
        outputs = model(**inputs)

    # softmax를 사용해 확률 계산
    probs = torch.softmax(
        outputs.logits,
        dim=1
    )

    # 가장 높은 확률의 클래스 선택
    prediction = torch.argmax(
        probs,
        dim=1
    ).item()

    # confidence 추출
    confidence = probs[0][prediction].item()

    # 예측 결과 변환
    # 1 = 긴급
    # 0 = 비긴급
    if prediction == 1:
        result = "urgent"

    else:
        result = "not_urgent"

    # JSON 형태로 결과 반환
    return {

        # 숫자 라벨
        "label": prediction,

        # 문자열 결과
        "result": result,

        # 예측 확률
        "confidence": round(confidence, 4)
    }