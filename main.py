from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# 1. CORS 설정: 요청을 보낸 Vercel 프론트엔드 주소를 정확히 등록 (끝에 / 없음)
origins = [
    "https://vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # 브라우저 preflight용 OPTIONS 필수
    allow_headers=["*"],                       # 모든 헤더 허용
)

# 테스트용 데이터 모델
class PredictInput(BaseModel):
    data: str

# main.py 파일 수정 (프론트엔드 주소에 맞춤)
@app.post("/api/epitope/predict/")
async def predict(item: PredictInput):
    # 기존 예측 로직
    return {"status": "success", "received": item.data}

@app.get("/")
def read_root():
    return {"message": "ImmuneNexus API Server is running!"}
