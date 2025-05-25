from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import base64  # Add this import for decoding base64 images

from app.services.diagnostic_service import DiagnosticService
from app.services.multiview_service import MultiViewDiagnosticService
from .models import ModelLoader
from .schemas import PredictionResponse, FinalResponse
import os
from fastapi import UploadFile, File, Form

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

@app.post("/upload_multiview_mri/")
async def upload_multiview_mri(
    axial: UploadFile = File(...),
    coronal: UploadFile = File(...),
    sagittal: UploadFile = File(...),
    user_id: str = Form("default_user")
):
    # Create temporary files for each view
    temp_files = {}
    try:
        for view, upload_file in {
            "axial": axial,
            "coronal": coronal,
            "sagittal": sagittal
        }.items():
            content = await upload_file.read()
            temp_file_path = f"temp_{view}_image.png"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(content)
            temp_files[view] = temp_file_path
        
        # Initialize MultiViewDiagnosticService
        diagnostic_service = MultiViewDiagnosticService(user_id=user_id)
        
        # Process the images
        response = diagnostic_service.process_multiview(temp_files)
        
        # Return the FinalResponse
        return response.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary files
        for temp_file in temp_files.values():
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
                
@app.post("/process_multiview_mri/")
async def process_multiview_mri(
    payload: dict
):
    # Extract and validate fields from the payload
    axial_files_base64 = payload.get("axial")
    coronal_files_base64 = payload.get("coronal")
    sagittal_files_base64 = payload.get("sagittal")
    user_id = payload.get("user_id")
    
    if not all([axial_files_base64, coronal_files_base64, sagittal_files_base64]):
        raise HTTPException(status_code=400, detail="All three MRI views (axial, coronal, sagittal) are required.")
    
    # Ensure each view is a list
    if not isinstance(axial_files_base64, list):
        axial_files_base64 = [axial_files_base64]
    if not isinstance(coronal_files_base64, list):
        coronal_files_base64 = [coronal_files_base64]
    if not isinstance(sagittal_files_base64, list):
        sagittal_files_base64 = [sagittal_files_base64]
    
    # Create temporary files for each view
    temp_files = {"axial": [], "coronal": [], "sagittal": []}
    try:
        # Process axial files
        for i, file_base64 in enumerate(axial_files_base64):
            try:
                file_data = base64.b64decode(file_base64)
                temp_file_path = f"temp_axial_{i}_image.png"
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(file_data)
                temp_files["axial"].append(temp_file_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image data for axial view (image {i+1}): {str(e)}")
        
        # Process coronal files
        for i, file_base64 in enumerate(coronal_files_base64):
            try:
                file_data = base64.b64decode(file_base64)
                temp_file_path = f"temp_coronal_{i}_image.png"
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(file_data)
                temp_files["coronal"].append(temp_file_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image data for coronal view (image {i+1}): {str(e)}")
        
        # Process sagittal files
        for i, file_base64 in enumerate(sagittal_files_base64):
            try:
                file_data = base64.b64decode(file_base64)
                temp_file_path = f"temp_sagittal_{i}_image.png"
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(file_data)
                temp_files["sagittal"].append(temp_file_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image data for sagittal view (image {i+1}): {str(e)}")
        
        # Initialize MultiViewDiagnosticService
        diagnostic_service = MultiViewDiagnosticService(user_id=user_id)
        
        # Process the images
        response = diagnostic_service.process_multiview(temp_files)
        
        # Return the FinalResponse
        return response.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary files
        for view_files in temp_files.values():
            for temp_file in view_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

@app.get("/")
async def process_mri():
    return {"message": "Welcome to the MRI Diagnostic API!"}



