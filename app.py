import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from infer_onnx import EdgeCaptionEngine

# Shared runtime instance loaded across application lifetime
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Load ONNX graph and tokenizer once into memory at startup
    engine = EdgeCaptionEngine()
    yield
    engine = None

app = FastAPI(
    title="Edge Image Captioning API",
    description="Low-latency vision-language microservice running INT8 quantized ONNX",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_loaded": engine is not None}

@app.post("/api/v1/caption")
async def create_caption(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        caption = engine.generate_caption(image)
        
        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "caption": caption
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)