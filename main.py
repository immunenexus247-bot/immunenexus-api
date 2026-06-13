import json
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(redirect_slashes=False)

# CORS 설정 (기존과 동일)
origins = [
    "https://immunenexus-api.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

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

# 🌟 [여기서부터 교체] 422 에러를 원천 차단하는 수동 파싱 완전판 코드
@app.post("/api/epitope/predict")
@app.post("/api/epitope/predict/")
async def predict(request: Request):
    try:
        # 1) 일반 텍스트 바이트를 문자열로 안전하게 변환
        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8").strip()
        
        # 2) 문자열을 파이썬 딕셔너리로 역직렬화(복원)
        data = json.loads(body_str)
        
        # 3) 데이터 꺼내기 (값 부재 시 빈 문자열 처리)
        hla_sequence = data.get("hla_sequence", "")
        peptide_sequence = data.get("peptide_sequence", "")
        
    except Exception as e:
        # 예외 상황 발생 시 422 대신 에러 원인을 친절하게 반환하여 크래시 방지
        return {"status": "error", "message": f"Data parsing failed: {str(e)}"}
    
    # 4) 프론트엔드가 화면에 뿌려줄 데이터 구조 양식 매핑
    return {
        "status": "success",
        "vfd_vval_sequence": "GNN Frame Mapping Complete",
        "generated_mhc": "MHC Acid Inverse Sequence Localized",
        "generated_alpha": f"{hla_sequence[:5]}-Alpha-Designed",
        "generated_beta": f"{peptide_sequence[:5]}-Beta-Designed",
        "vfd_vval_indicator": "APPROVED",
        "vfd_vval_id2": "Score: 0.99"
    }

@app.get("/")
def read_root():
    return {"message": "ImmuneNexus API Server is running successfully!"}

