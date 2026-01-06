from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import shutil
import os
import uuid
import json
from typing import List
from audio_redaction_service import AudioRedactor

app = FastAPI()

# Ensure directories exist
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Redactor
# Set use_mock_transcription=False for REAL Whisper transcription
# Note: Real transcription will be slower but will actually transcribe your audio
redactor = AudioRedactor(use_mock_transcription=False)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/redact")
async def redact_audio_endpoint(
    file: UploadFile = File(...),
    entities: str = Form(""), # Comma separated list
    mode: str = Form("beep"),
    denoise: bool = Form(False),
    denoise_intensity: float = Form(0.8),
    detection_mode: str = Form("gemini")  # 'regex', 'gemini', or 'ai'
):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    output_filename = f"redacted_{file.filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    entity_list = [e.strip() for e in entities.split(",") if e.strip()]
    
    # Define generator for streaming response
    def processing_generator():
        try:
            # Yield updates from the pipeline
            for update in redactor.process_pipeline_stream(input_path, output_path, entity_list, mode, denoise, denoise_intensity, detection_mode):
                # Format as NDJSON (Newline Delimited JSON)
                yield json.dumps(update) + "\n"
        except Exception as e:
            # Yield error
            yield json.dumps({"status": "error", "error": str(e)}) + "\n"
            
    return StreamingResponse(processing_generator(), media_type="application/x-ndjson")

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        media_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        return FileResponse(file_path, media_type=media_type, filename=filename)
    return JSONResponse(content={"error": "File not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    print("Starting Audio Redaction Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
