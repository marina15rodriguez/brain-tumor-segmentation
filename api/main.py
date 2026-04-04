"""FastAPI application for brain tumour segmentation.

Endpoints:
    GET  /           — interactive web UI
    GET  /health     — check the server and model are ready
    POST /predict    — upload an MRI slice, get back a segmentation mask

Usage:
    cd api
    uvicorn main:app --reload

Then open http://localhost:8000 in your browser for the visual interface.
"""

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

try:
    from predict import load_model, run_prediction          # when run from api/
except ImportError:
    from api.predict import load_model, run_prediction      # when run from project root (Docker)


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
@app.get("/", response_class=HTMLResponse)
def root():
    """Interactive web UI for tumour segmentation."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain Tumour Segmentation</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f1117;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }

        h1 {
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 6px;
            color: #ffffff;
        }

        .subtitle {
            color: #888;
            font-size: 0.95rem;
            margin-bottom: 40px;
        }

        /* Drop zone */
        #dropzone {
            width: 100%;
            max-width: 480px;
            border: 2px dashed #444;
            border-radius: 12px;
            padding: 48px 24px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
            background: #1a1d27;
        }

        #dropzone:hover, #dropzone.dragover {
            border-color: #4f8ef7;
            background: #1e2236;
        }

        #dropzone .icon { font-size: 2.5rem; margin-bottom: 12px; }
        #dropzone p { color: #aaa; font-size: 0.9rem; line-height: 1.6; }
        #dropzone span { color: #4f8ef7; }
        #fileInput { display: none; }

        /* Spinner */
        #spinner {
            display: none;
            margin: 32px auto;
            width: 40px; height: 40px;
            border: 4px solid #333;
            border-top-color: #4f8ef7;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Results */
        #results { display: none; width: 100%; max-width: 900px; margin-top: 36px; }

        .badge {
            display: inline-block;
            padding: 6px 18px;
            border-radius: 20px;
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 24px;
        }
        .badge.positive { background: #3a1f1f; color: #f87171; border: 1px solid #f87171; }
        .badge.negative { background: #1a2e1a; color: #4ade80; border: 1px solid #4ade80; }

        .stats {
            background: #1a1d27;
            border-radius: 10px;
            padding: 16px 24px;
            margin-bottom: 24px;
            font-size: 0.9rem;
            color: #aaa;
        }
        .stats strong { color: #fff; }

        .images {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 16px;
        }

        .img-card {
            background: #1a1d27;
            border-radius: 10px;
            overflow: hidden;
            text-align: center;
        }

        .img-card p {
            padding: 10px;
            font-size: 0.82rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .img-card img {
            width: 100%;
            display: block;
            image-rendering: pixelated;
        }

        /* Reset button */
        #resetBtn {
            display: none;
            margin-top: 28px;
            padding: 10px 28px;
            background: #2a2d3a;
            color: #ccc;
            border: 1px solid #444;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.2s;
        }
        #resetBtn:hover { background: #3a3d4a; }

        #error {
            display: none;
            color: #f87171;
            background: #2a1a1a;
            border: 1px solid #f87171;
            border-radius: 8px;
            padding: 12px 20px;
            margin-top: 20px;
            font-size: 0.9rem;
            max-width: 480px;
            width: 100%;
            text-align: center;
        }

        @media (max-width: 600px) {
            .images { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

    <h1>Brain Tumour Segmentation</h1>
    <p class="subtitle">Upload an MRI slice — the model will highlight the tumour region</p>

    <div id="dropzone" onclick="document.getElementById('fileInput').click()">
        <div class="icon">🧠</div>
        <p>Drag & drop an MRI slice here<br>or <span>click to browse</span></p>
        <p style="margin-top:8px; font-size:0.8rem;">Supported: TIFF, PNG, JPEG</p>
        <input type="file" id="fileInput" accept=".tif,.tiff,.png,.jpg,.jpeg">
    </div>

    <div id="spinner"></div>
    <div id="error"></div>

    <div id="results">
        <div id="badge"></div>
        <div class="stats" id="stats"></div>
        <div class="images">
            <div class="img-card">
                <p>Original</p>
                <img id="imgOriginal">
            </div>
            <div class="img-card">
                <p>Predicted mask</p>
                <img id="imgMask">
            </div>
            <div class="img-card">
                <p>Overlay</p>
                <img id="imgOverlay">
            </div>
        </div>
    </div>

    <button id="resetBtn" onclick="reset()">Upload another image</button>

    <script>
        const dropzone  = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const spinner   = document.getElementById('spinner');
        const results   = document.getElementById('results');
        const errorDiv  = document.getElementById('error');
        const resetBtn  = document.getElementById('resetBtn');

        // Drag and drop
        dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
        dropzone.addEventListener('drop', e => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

        function handleFile(file) {
            if (!file) return;

            // Show spinner, hide everything else
            dropzone.style.display  = 'none';
            spinner.style.display   = 'block';
            results.style.display   = 'none';
            errorDiv.style.display  = 'none';
            resetBtn.style.display  = 'none';

            const formData = new FormData();
            formData.append('file', file);

            fetch('/predict', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    spinner.style.display = 'none';

                    if (data.detail) {
                        showError(data.detail);
                        return;
                    }

                    // Badge
                    const badge = document.getElementById('badge');
                    if (data.tumour_detected) {
                        badge.textContent = '⚠ Tumour detected';
                        badge.className = 'badge positive';
                    } else {
                        badge.textContent = '✓ No tumour detected';
                        badge.className = 'badge negative';
                    }

                    // Stats
                    document.getElementById('stats').innerHTML =
                        `File: <strong>${file.name}</strong> &nbsp;|&nbsp; ` +
                        `Tumour coverage: <strong>${(data.tumour_fraction * 100).toFixed(2)}%</strong> of the slice`;

                    // Images — all returned as base64 PNGs from the server
                    document.getElementById('imgOriginal').src = 'data:image/png;base64,' + data.original_png;
                    document.getElementById('imgMask').src     = 'data:image/png;base64,' + data.mask_png;
                    document.getElementById('imgOverlay').src  = 'data:image/png;base64,' + data.overlay_png;

                    results.style.display  = 'block';
                    resetBtn.style.display = 'inline-block';
                })
                .catch(() => {
                    spinner.style.display = 'none';
                    showError('Something went wrong. Is the server running?');
                });
        }

        function showError(msg) {
            errorDiv.textContent   = msg;
            errorDiv.style.display = 'block';
            dropzone.style.display = 'block';
        }

        function reset() {
            dropzone.style.display = 'block';
            results.style.display  = 'none';
            resetBtn.style.display = 'none';
            fileInput.value        = '';
        }
    </script>
</body>
</html>
"""


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
