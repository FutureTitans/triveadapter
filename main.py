"""
main.py — FastAPI application for NeuroViral.

Serves the brain analysis and script generation API endpoints,
and optionally serves the built frontend as static files.
"""

import os
import logging
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import hmac

from analyzer import analyze_content
from scripter import generate_script

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "neuroviral_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

APP_USERNAME = os.getenv("APP_USERNAME")
APP_PASSWORD = os.getenv("APP_PASSWORD")
AUTH_ENABLED = bool(APP_USERNAME and APP_PASSWORD)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

def verify_token(token: Optional[str] = Depends(oauth2_scheme)):
    if not AUTH_ENABLED:
        return True
    if not token or not hmac.compare_digest(token, APP_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return True

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroViral API",
    description="Brain-encoded content analysis powered by Meta TRIBE v2",
    version="1.0.0",
)

# CORS — allow frontend dev server and common origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ───────────────────────────────────────────────────────────────────
class ScriptRequest(BaseModel):
    roi_scores: dict
    viral_score: float
    peak_moments: list
    dominant_modality: str
    content_type: str = "Short-form"
    tone: str = "Educational"


class AnalysisResponse(BaseModel):
    brain_data: list
    timeline: list
    peak_moments: list
    roi_scores: dict
    viral_score: float
    dominant_modality: str


class ScriptResponse(BaseModel):
    hooks: list
    script: str
    headline: str
    caption: str
    tips: list


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model": "TRIBE v2", "auth_enabled": AUTH_ENABLED}

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not AUTH_ENABLED:
        return {"access_token": "dummy", "token_type": "bearer"}
    
    if not (hmac.compare_digest(form_data.username, APP_USERNAME) and 
            hmac.compare_digest(form_data.password, APP_PASSWORD)):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    # Using the password as a simple stateless session token (secure over HTTPS)
    return {"access_token": APP_PASSWORD, "token_type": "bearer"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    _: bool = Depends(verify_token)
):
    """
    Analyze content using TRIBE v2 brain encoding model.
    Accepts either a file upload (video/audio) or raw text.
    """
    if file is None and text is None:
        raise HTTPException(
            status_code=400,
            detail="Please provide either a file or text to analyze.",
        )

    file_path = None

    try:
        if file is not None:
            # Validate file type
            allowed_extensions = {".mp4", ".mov", ".mp3", ".wav", ".txt"}
            ext = Path(file.filename).suffix.lower()
            if ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}",
                )

            # Save uploaded file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            result = await analyze_content(file_path=file_path)
        else:
            if not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Text input cannot be empty.",
                )
            result = await analyze_content(text=text)

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up uploaded file
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)


@app.post("/generate-script", response_model=ScriptResponse)
async def create_script(request: ScriptRequest, _: bool = Depends(verify_token)):
    """
    Generate a viral script using AI based on brain analysis results.
    """
    try:
        result = await generate_script(
            roi_scores=request.roi_scores,
            viral_score=request.viral_score,
            peak_moments=request.peak_moments,
            dominant_modality=request.dominant_modality,
            content_type=request.content_type,
            tone=request.tone,
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Script generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Script generation failed: {str(e)}"
        )


# ── Serve Frontend (production) ─────────────────────────────────────────────
FRONTEND_BUILD = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(FRONTEND_BUILD):
    app.mount("/", StaticFiles(directory=FRONTEND_BUILD, html=True), name="frontend")


# ── Startup ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
