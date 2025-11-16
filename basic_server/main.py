import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from utils.ocr_service import verify_document
from utils.validators import validate_aadhaar, validate_pan, validate_driving_license

load_dotenv()

app = FastAPI(
    title="Document Verification API",
    description="FastAPI server for document verification using Google Cloud Vision OCR",
    version="1.0.0"
)

PORT = int(os.getenv('PORT', 3000))
UPLOAD_DIR = Path('./uploads')
UPLOAD_DIR.mkdir(exist_ok=True)

# File validation
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'pdf']
ALLOWED_MIME_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'application/pdf']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def cleanup_file(file_path: Path):
    """Remove file if it exists"""
    if file_path.exists():
        file_path.unlink()


def error_response(message: str):
    """Generate standardized error response"""
    return {
        "errorMessage": message,
        "requiredExtensionType": ALLOWED_EXTENSIONS,
        "requiredFileType": "Valid document image or PDF"
    }


async def process_document(file: UploadFile) -> Path:
    """Save and validate uploaded file"""
    if not file:
        raise HTTPException(status_code=400, detail=error_response("No file uploaded"))
    
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=error_response(f"Invalid file type: {file.content_type}"))
    
    # Save file
    file_path = UPLOAD_DIR / file.filename
    content = await file.read()
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=error_response("File too large (max 10MB)"))
    
    with open(file_path, 'wb') as f:
        f.write(content)
    
    return file_path


@app.post("/verify-aadhaar")
async def verify_aadhaar_endpoint(document: UploadFile = File(...)):
    """Verify Aadhaar card document"""
    file_path = None
    try:
        file_path = await process_document(document)
        extracted_text = await verify_document(str(file_path))
        validation = await validate_aadhaar(extracted_text)
        cleanup_file(file_path)
        
        if validation['isValid']:
            return {
                "success": True,
                "message": "Aadhaar verified",
                "documentType": "Aadhaar",
                "details": validation['details']
            }
        
        raise HTTPException(
            status_code=400,
            detail=error_response(validation.get('error', 'Invalid Aadhaar card'))
        )
    except HTTPException:
        if file_path:
            cleanup_file(file_path)
        raise
    except Exception as e:
        if file_path:
            cleanup_file(file_path)
        raise HTTPException(status_code=500, detail=error_response(str(e)))


@app.post("/verify-pan")
async def verify_pan_endpoint(document: UploadFile = File(...)):
    """Verify PAN card document"""
    file_path = None
    try:
        file_path = await process_document(document)
        extracted_text = await verify_document(str(file_path))
        validation = await validate_pan(extracted_text)
        cleanup_file(file_path)
        
        if validation['isValid']:
            return {
                "success": True,
                "message": "PAN verified",
                "documentType": "PAN",
                "details": validation['details']
            }
        
        raise HTTPException(
            status_code=400,
            detail=error_response(validation.get('error', 'Invalid PAN card'))
        )
    except HTTPException:
        if file_path:
            cleanup_file(file_path)
        raise
    except Exception as e:
        if file_path:
            cleanup_file(file_path)
        raise HTTPException(status_code=500, detail=error_response(str(e)))


@app.post("/verify-driving-license")
async def verify_driving_license_endpoint(document: UploadFile = File(...)):
    """Verify Driving License document"""
    file_path = None
    try:
        file_path = await process_document(document)
        extracted_text = await verify_document(str(file_path))
        validation = await validate_driving_license(extracted_text)
        cleanup_file(file_path)
        
        if validation['isValid']:
            return {
                "success": True,
                "message": "License verified",
                "documentType": "Driving License",
                "details": validation['details']
            }
        
        raise HTTPException(
            status_code=400,
            detail=error_response(validation.get('error', 'Invalid driving license'))
        )
    except HTTPException:
        if file_path:
            cleanup_file(file_path)
        raise
    except Exception as e:
        if file_path:
            cleanup_file(file_path)
        raise HTTPException(status_code=500, detail=error_response(str(e)))


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Document Verification API is running"}


if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Server starting on http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
