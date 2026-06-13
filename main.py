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

# 2. 에러가 나던 predict 엔드포인트
@app.post("/predict/")
async def predict(item: PredictInput):
    # 이곳에 기존 예측(predict) 로직을 넣으시면 됩니다.
    return {"status": "success", "received": item.data}

@app.get("/")
def read_root():
    return {"message": "ImmuneNexus API Server is running!"}
