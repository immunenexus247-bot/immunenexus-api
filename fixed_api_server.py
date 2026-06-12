"""
ImmuneNexus™ Enterprise AI API Server
CORS Preflight & Redirect 오류 수정
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# ============================================================================
# 1. CORS 설정 (Preflight 오류 해결)
# ============================================================================

app = FastAPI(
    title="ImmuneNexus Enterprise AI API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS 미들웨어 추가 - OPTIONS 요청 자동 처리
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://yourdomain.com",  # 프로덕션 도메인으로 변경
        "*"  # 개발 환경에서만 사용 (프로덕션에선 제거)
    ],
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, PUT, DELETE, OPTIONS 등 모두 허용
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Accept",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
    max_age=3600,  # preflight 캐시 1시간
)

# ============================================================================
# 2. Static Files 설정 (리다이렉트 오류 해결)
# ============================================================================

# static 폴더 존재 확인 및 생성
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Static files를 "/static" 경로에 마운트
try:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
except RuntimeError as e:
    print(f"⚠️  Static files 마운트 경고: {e}")

# ============================================================================
# 3. 리다이렉트 처리
# ============================================================================

# https://example.com/ → https://example.com (trailing slash 제거)
# 또는 반대로 처리 필요 시 활성화
@app.middleware("http")
async def redirect_trailing_slash(request, call_next):
    """
    Trailing slash 리다이렉트 미들웨어
    비활성화하려면 return await call_next(request)로 변경
    """
    if request.url.path != "/" and request.url.path.endswith("/"):
        # /api/endpoint/ → /api/endpoint
        new_url = request.url.remove_query_params().path.rstrip("/")
        return RedirectResponse(url=new_url, status_code=301)

    return await call_next(request)

# ============================================================================
# 4. API 라우트 예시 (올바른 경로 설정)
# ============================================================================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "status": "online",
        "service": "ImmuneNexus Enterprise AI API",
        "api_docs": "/api/docs"
    }

@app.get("/api/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {"status": "healthy", "service": "ImmuneNexus"}

@app.post("/api/epitope/predict")
async def predict_epitope(data: dict):
    """
    에피토프 예측 엔드포인트

    요청 예시:
    {
        "hla_sequence": "GSHSMRYFYTAVT...",
        "peptide_sequence": "SIINFEKL",
        "model_type": "tcr_inference"
    }
    """
    if not data.get("hla_sequence") or not data.get("peptide_sequence"):
        raise HTTPException(status_code=400, detail="Missing required fields")

    return {
        "status": "success",
        "prediction": 0.87,
        "confidence": "high"
    }

# ============================================================================
# 5. Render 플랫폼 배포 설정 (Sleep mode 해결)
# ============================================================================

import threading
import time

def keep_alive():
    """
    Render 무료 플랜 sleep mode 방지
    Periodic health check를 자체 호스트에서 실행
    """
    def ping_self():
        while True:
            time.sleep(600)  # 10분마다 실행
            try:
                import requests
                port = os.getenv("PORT", "8000")
                requests.get(f"http://localhost:{port}/api/health", timeout=5)
                print("✅ Keep-alive ping sent")
            except Exception as e:
                print(f"⚠️  Keep-alive failed: {e}")

    # 별도 스레드에서 실행
    thread = threading.Thread(target=ping_self, daemon=True)
    thread.start()

# ============================================================================
# 6. 서버 실행
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))

    # Keep-alive 활성화 (Render 배포 시)
    keep_alive()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,  # Render 무료 플랜에서는 1개 권장
        access_log=True,
        log_level="info"
    )
