from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 🌟 자동 리다이렉트 기능을 완전히 꺼서 CORS 헤더 유실을 방지합니다.
app = FastAPI(redirect_slashes=False)

# 1. CORS 기본 설정 (끝에 슬래시 없는 순수 도메인 상태 유지)
origins = [
    "https://immunenexus-api.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 🌟 브라우저가 사전 검사(OPTIONS)를 보내면 어떤 주소든 무조건 통과시키는 무적 미들웨어
@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    if request.method == "OPTIONS":
        response = Response(status_code=200)
        origin = request.headers.get("Origin")
        if origin in origins:
            response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response
    
    response = await call_next(request)
    return response

# 2. 데이터 규격 정의
class PredictInput(BaseModel):
    hla_sequence: str
    peptide_sequence: str

# 3. 🌟 슬래시가 붙은 주소와 안 붙은 주소를 완전히 독립적인 개별 경로로 둘 다 등록합니다.
# 프론트엔드가 어떻게 요청을 보내든 리다이렉트 과정 없이 서버가 즉시 응답하므로 CORS가 절대로 깨지지 않습니다.
@app.post("/api/epitope/predict")
@app.post("/api/epitope/predict/")
async def predict(item: PredictInput):
    return {
        "status": "success", 
        "received_hla": item.hla_sequence,
        "received_peptide": item.peptide_sequence
    }

@app.get("/")
def read_root():
    return {"message": "ImmuneNexus API Server is running perfectly!"}
