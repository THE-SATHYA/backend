from fastapi import FastAPI, File, UploadFile
import torch
import librosa
import numpy as np
import io
from model_loader import load_model, preprocess_audio
from fastapi.responses import FileResponse
import os
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Load trained model
model = load_model("best_model.pth")

origins = [
    "https://frontend-three-ecru-71.vercel.app/",  
    "http://127.0.0.1.10000" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST, GET"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Music Classification API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded audio file
        audio_data = await file.read()
        
        # Preprocess audio
        mfcc_tensor = preprocess_audio(audio_data)
        
        # Make prediction
        with torch.no_grad():
            output = model(mfcc_tensor)
            pred_idx = output.argmax(dim=1).item()
            label_map = {0: "flow", 1: "breathy", 2: "neutral", 3: "pressed"}
            predicted_label = label_map.get(pred_idx, "unknown")
        
        return {"filename": file.filename, "predicted_class": predicted_label}
    
    except Exception as e:
        return {"error": str(e)}

from fastapi import FastAPI, UploadFile, File

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename, "status": "uploaded successfully"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve a favicon if available."""
    if os.path.exists("favicon.ico"):
        return FileResponse("favicon.ico")
    return {"message": "No favicon found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=10000)
