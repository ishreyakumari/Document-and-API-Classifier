# Document Verification API - FastAPI

FastAPI server for document verification using Google Cloud Vision OCR and Google Gemini AI.

## Features

- ✅ **Aadhaar Card Verification** - `/verify-aadhaar`
- ✅ **PAN Card Verification** - `/verify-pan`
- ✅ **Driving License Verification** - `/verify-driving-license`
- 🤖 **AI-Enhanced Validation** using Google Gemini
- 📄 **PDF Support** - Automatically converts PDF to image for OCR
- 🧪 **Mock Mode** - Test without Google Cloud credentials

## Requirements

- Python 3.8+
- Google Cloud Vision API credentials (or use Mock Mode)
- Poppler (for PDF processing)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Poppler (for PDF processing)

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases/)

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:
```env
PORT=3000
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json
GEMINI_API_KEY=your_gemini_api_key_here
MOCK_OCR=false
```

### 4. Google Cloud Vision Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable Vision API
3. Create a service account and download JSON key
4. Save as `service-account-key.json` in project root

### 5. Google Gemini API (Optional)

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Running the Server

### Development Mode

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --port 3000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 3000 --workers 4
```

Server will start at: `http://localhost:3000`

## API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:3000/docs
- **ReDoc**: http://localhost:3000/redoc

## API Endpoints

### 1. Verify Aadhaar Card

```bash
POST /verify-aadhaar
Content-Type: multipart/form-data

curl -X POST "http://localhost:3000/verify-aadhaar" \
  -F "document=@aadhaar.jpg"
```

### 2. Verify PAN Card

```bash
POST /verify-pan
Content-Type: multipart/form-data

curl -X POST "http://localhost:3000/verify-pan" \
  -F "document=@pan.pdf"
```

### 3. Verify Driving License

```bash
POST /verify-driving-license
Content-Type: multipart/form-data

curl -X POST "http://localhost:3000/verify-driving-license" \
  -F "document=@license.png"
```

## Response Format

### Success Response

```json
{
  "success": true,
  "message": "Aadhaar verified",
  "documentType": "Aadhaar",
  "details": {}
}
```

### Error Response

```json
{
  "errorMessage": "Invalid Aadhaar card",
  "requiredExtensionType": ["jpg", "jpeg", "png", "pdf"],
  "requiredFileType": "Valid document image or PDF"
}
```

## File Constraints

- **Allowed formats**: JPG, JPEG, PNG, PDF
- **Max file size**: 10MB
- **PDF**: Only first page is processed

## Mock Mode (Testing)

Test without Google Cloud credentials:

```env
MOCK_OCR=true
```

Mock mode returns sample text based on filename:
- Files with "aadhaar" → Mock Aadhaar text
- Files with "pan" → Mock PAN text
- Files with "license" → Mock License text

## Project Structure

```
basic_server/
├── main.py                    # FastAPI application
├── utils/
│   ├── ocr_service.py        # Google Cloud Vision OCR
│   └── validators.py         # Document validation logic
├── uploads/                  # Temporary file storage
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create from .env.example)
├── .env.example             # Environment template
└── service-account-key.json # Google Cloud credentials
```

## Migration from Node.js

This codebase has been converted from Express.js to FastAPI:

- `server.js` → `main.py`
- `utils/ocrService.js` → `utils/ocr_service.py`
- `utils/validators.js` → `utils/validators.py`
- `package.json` → `requirements.txt`

## Troubleshooting

### PDF Processing Error

If you get "poppler not found":
- Ensure poppler is installed (see Installation step 2)
- On macOS: `brew install poppler`
- On Linux: `sudo apt-get install poppler-utils`

### Vision API Error

If you get "Vision API not initialized":
- Check `GOOGLE_APPLICATION_CREDENTIALS` path
- Verify service account key is valid
- Enable Vision API in Google Cloud Console
- Or use `MOCK_OCR=true` for testing

### Import Errors

```bash
pip install -r requirements.txt --upgrade
```

## License

ISC
