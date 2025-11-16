import os
import re
from typing import Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

model = None

if os.getenv('GEMINI_API_KEY'):
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-2.0-flash')
    print('✅ Gemini AI initialized')


async def validate_with_ai(text: str, document_type: str) -> Dict[str, Any]:
    """
    Validate document using Gemini AI
    
    Args:
        text: OCR extracted text
        document_type: Type of document ('aadhaar', 'pan', 'driving-license')
    
    Returns:
        Dictionary with validation results
    """
    if not model:
        raise Exception('AI not available')
    
    names = {
        'aadhaar': 'Indian Aadhaar Card',
        'pan': 'Indian PAN Card',
        'driving-license': 'Indian Driving License'
    }
    
    prompt = f"""Is this a valid {names[document_type]}?

OCR Text:
{text}

Answer ONLY:
ANSWER: YES/NO
REASON: brief explanation"""
    
    response = model.generate_content(prompt)
    ai_text = response.text.strip()
    
    is_valid = bool(re.search(r'ANSWER:\s*YES', ai_text, re.IGNORECASE))
    reason_match = re.search(r'REASON:\s*(.+)', ai_text, re.IGNORECASE)
    reason = reason_match.group(1) if reason_match else 'No reason'
    
    print(f"🤖 AI: {'YES' if is_valid else 'NO'} - {reason}")
    
    return {
        'isValid': is_valid,
        'reason': reason,
        'details': {}
    }


async def validate_aadhaar(text: str) -> Dict[str, Any]:
    """
    Validate Aadhaar card from OCR text
    
    Args:
        text: OCR extracted text
    
    Returns:
        Dictionary with validation results
    """
    if model:
        try:
            return await validate_with_ai(text, 'aadhaar')
        except Exception as error:
            print(f'⚠️  AI failed, using fallback: {error}')
    
    # Fallback validation
    normalized = text.lower()
    has_keyword = bool(re.search(r'(aadhaar|aadhar|uidai|आधार)', text, re.IGNORECASE))
    has_number = bool(re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text))
    
    is_valid = has_keyword or has_number
    
    return {
        'isValid': is_valid,
        'error': None if is_valid else 'Not an Aadhaar card',
        'details': {}
    }


async def validate_pan(text: str) -> Dict[str, Any]:
    """
    Validate PAN card from OCR text
    
    Args:
        text: OCR extracted text
    
    Returns:
        Dictionary with validation results
    """
    if model:
        try:
            return await validate_with_ai(text, 'pan')
        except Exception as error:
            print(f'⚠️  AI failed, using fallback: {error}')
    
    # Fallback validation
    has_keyword = bool(re.search(r'(income tax|permanent account|pan|आयकर)', text, re.IGNORECASE))
    has_number = bool(re.search(r'\b[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]\b', text, re.IGNORECASE))
    
    is_valid = has_keyword or has_number
    
    return {
        'isValid': is_valid,
        'error': None if is_valid else 'Not a PAN card',
        'details': {}
    }


async def validate_driving_license(text: str) -> Dict[str, Any]:
    """
    Validate Driving License from OCR text
    
    Args:
        text: OCR extracted text
    
    Returns:
        Dictionary with validation results
    """
    if model:
        try:
            return await validate_with_ai(text, 'driving-license')
        except Exception as error:
            print(f'⚠️  AI failed, using fallback: {error}')
    
    # Fallback validation
    has_keyword = bool(re.search(r'(driving licen[cs]e|transport|rto|motor vehicle)', text, re.IGNORECASE))
    has_number = bool(re.search(r'\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}\b', text, re.IGNORECASE))
    
    is_valid = has_keyword or has_number
    
    return {
        'isValid': is_valid,
        'error': None if is_valid else 'Not a driving license',
        'details': {}
    }
