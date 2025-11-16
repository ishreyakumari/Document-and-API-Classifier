import os
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import vision
from pdf2image import convert_from_path
import io

# Load environment variables
load_dotenv()

MOCK_MODE = os.getenv('MOCK_OCR', 'false').lower() == 'true'
client = None

if not MOCK_MODE:
    try:
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path:
            # Convert relative path to absolute path
            abs_credentials_path = os.path.abspath(credentials_path)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = abs_credentials_path
            
            if os.path.exists(abs_credentials_path):
                client = vision.ImageAnnotatorClient()
                print(f'✅ Google Cloud Vision initialized with credentials: {abs_credentials_path}')
            else:
                print(f'❌ Credentials file not found: {abs_credentials_path}')
        else:
            print('❌ GOOGLE_APPLICATION_CREDENTIALS not set')
    except Exception as error:
        print(f'❌ Vision API error: {error}')
else:
    print('🧪 MOCK MODE enabled')


def mock_ocr(file_path: str) -> str:
    """Mock OCR for testing without Google Cloud credentials"""
    filename = Path(file_path).name.lower()
    
    if 'aadhaar' in filename or 'aadhar' in filename:
        return 'GOVERNMENT OF INDIA\nआधार\nUIDAI\n1234 5678 9012\nName: John Doe'
    elif 'pan' in filename:
        return 'INCOME TAX DEPARTMENT\nPermanent Account Number\nABCDE1234F'
    elif 'license' in filename or 'licence' in filename:
        return 'DRIVING LICENCE\nDL-1420110012345\nName: JOHN DOE'
    
    return 'SAMPLE DOCUMENT'


async def verify_document(file_path: str) -> str:
    """
    Extract text from document using Google Cloud Vision OCR
    
    Args:
        file_path: Path to the document file (image or PDF)
    
    Returns:
        Extracted text from the document
    
    Raises:
        Exception: If file not found or OCR fails
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError('File not found')
        
        if MOCK_MODE:
            return mock_ocr(file_path)
        
        if not client:
            raise Exception('Vision API not initialized')
        
        file_ext = path.suffix.lower()
        
        # Handle PDF files
        if file_ext == '.pdf':
            try:
                # Convert first page of PDF to image
                images = convert_from_path(file_path, first_page=1, last_page=1, dpi=200)
                
                if images:
                    # Convert PIL Image to bytes
                    img_byte_arr = io.BytesIO()
                    images[0].save(img_byte_arr, format='PNG')
                    file_buffer = img_byte_arr.getvalue()
                else:
                    # Fallback to raw PDF
                    with open(file_path, 'rb') as f:
                        file_buffer = f.read()
            except Exception as e:
                print(f'⚠️  PDF conversion failed: {e}, using raw PDF')
                with open(file_path, 'rb') as f:
                    file_buffer = f.read()
        else:
            # Handle image files
            with open(file_path, 'rb') as f:
                file_buffer = f.read()
        
        # Perform OCR
        image = vision.Image(content=file_buffer)
        response = client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(f'Vision API error: {response.error.message}')
        
        # Extract text from response
        texts = response.text_annotations
        if texts:
            text = texts[0].description
        elif response.full_text_annotation:
            text = response.full_text_annotation.text
        else:
            text = None
        
        if not text:
            raise Exception('No text detected in document')
        
        return text
        
    except Exception as error:
        raise Exception(f'OCR failed: {str(error)}')
