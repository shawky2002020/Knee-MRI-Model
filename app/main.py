from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import base64  # Add this import for decoding base64 images

from app.services.diagnostic_service import DiagnosticService
from .models import ModelLoader
from .schemas import PredictionResponse, FinalResponse
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model loader
model_loader = ModelLoader()

@app.post("/process_mri/")
async def process_mri(
    payload: dict  # Accept JSON payload
):
    # Extract and validate fields from the payload
    file_base64 = payload.get("file")
    view_type = payload.get("view_type")
    user_id = payload.get("user_id")

    if not file_base64 or not view_type:
        raise HTTPException(status_code=400, detail="Both 'file' and 'view_type' are required.")

    # Decode the base64-encoded image and save it temporarily
    try:
        file_data = base64.b64decode(file_base64)
        temp_file_path = "temp_image.png"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

    try:
        # Initialize DiagnosticService with optional user_id
        diagnostic_service = DiagnosticService(user_id=user_id)

        # Process the image
        response = diagnostic_service.process_image(temp_file_path, view_type)

        # Return the FinalResponse
        return response.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary file
        os.remove(temp_file_path)

@app.get("/")
async def process_mri():
    return {"message": "Welcome to the MRI Diagnostic API!"}



