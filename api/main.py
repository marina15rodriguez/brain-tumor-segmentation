"""FastAPI application for brain tumour segmentation.

Endpoints:
    GET  /           — welcome message
    GET  /health     — check the server and model are ready
    POST /predict    — upload an MRI slice, get back a segmentation mask

Usage:
    cd api
    uvicorn main:app --reload

Then open http://localhost:8000/docs in your browser for an interactive
interface where you can test the API without writing any code.
"""

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from predict import load_model, run_prediction


# ---------------------------------------------------------------------------
# Startup: load model once when the server starts
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "results" / "best_model.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup, release resources at shutdown."""
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Train the model first with: python src/train.py"
        )
    load_model(CHECKPOINT_PATH)
    yield
    # Nothing to clean up for a simple torch model


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title       = "Brain Tumour Segmentation API",
    description = "Upload an MRI slice and get back a binary tumour mask.",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Welcome message."""
    return {
        "message": "Brain Tumour Segmentation API",
        "docs":    "Visit /docs for the interactive interface",
    }


@app.get("/health")
def health():
    """Check that the server is running and the model is loaded."""
    return {"status": "ok", "checkpoint": str(CHECKPOINT_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload an MRI slice image and receive a tumour segmentation.

    Accepts any common image format (TIFF, PNG, JPEG).
    Returns a JSON response with:
      - tumour_detected  : whether a tumour was found
      - tumour_fraction  : fraction of pixels predicted as tumour (0.0 to 1.0)
      - mask_png         : base64-encoded PNG of the binary mask (white = tumour)
      - overlay_png      : base64-encoded PNG of the MRI with tumour highlighted in red
    """
    # Validate file type
    allowed = {"image/tiff", "image/png", "image/jpeg", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: TIFF, PNG, JPEG.",
        )

    # Read uploaded bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Run inference
    try:
        result = run_prediction(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return JSONResponse(content=result)
