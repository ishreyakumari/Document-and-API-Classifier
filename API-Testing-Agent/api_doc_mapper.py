"""
API Document Mapper - Main Script

This script:
1. Pre-classifies ALL documents using Gemini Vision (OCR + AI) and creates a map
2. Reads a Postman collection and identifies file upload APIs
3. Tests each API with a random document from the map
4. Parses error responses to understand what document type the API requires
5. Automatically retries with the correct document type from the map
6. Generates a comprehensive report showing both failures and successes
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydantic import BaseModel, Field
from markdownify import markdownify as md
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GENAI_KEY = os.getenv("GOOGLE_API_KEY")
if not GENAI_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable. Please set it in .env file")
genai.configure(api_key=GENAI_KEY)

GEMINI_MODEL = "gemini-2.5-pro"


# ==================== DATA MODELS ====================

class ErrorStruct(BaseModel):
    """Standardized error structure returned by APIs"""
    required_extension_type: Optional[str] = Field(default=None)
    required_document_type: Optional[str] = Field(default=None)
    description: str


class DocClassification(BaseModel):
    """Document classification result from Gemini"""
    document_type: Optional[str] = None
    confidence: Optional[float] = None


class ApiResult(BaseModel):
    """Complete result for one API test"""
    api_name: str
    method: str
    url: str
    picked_file: Optional[str]
    local_file_classification: Optional[DocClassification] = None
    error_struct: Optional[ErrorStruct] = None
    raw_response: Dict[str, Any] = {}


# ==================== UTILITY FUNCTIONS ====================

def log(msg: str, end: str = "\n"):
    """Print timestamped log message"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", end=end)


def load_postman(path: str) -> Dict[str, Any]:
    """Load Postman collection or environment JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_env_vars(env_path: Optional[str]) -> Dict[str, str]:
    """Load Postman environment variables"""
    if not env_path or not os.path.exists(env_path):
        return {}
    
    data = load_postman(env_path)
    variables = {}
    for v in data.get("values", []):
        if not v.get("enabled", True):
            continue
        variables[v["key"]] = v.get("value", "")
    return variables


def resolve_postman_vars(s: str, env: Dict[str, str]) -> str:
    """Replace {{variable}} with actual values from environment"""
    def repl(m):
        key = m.group(1)
        return env.get(key, m.group(0))
    return re.sub(r"\{\{([^}]+)\}\}", repl, s)


def is_upload_request(item: Dict[str, Any]) -> bool:
    """Check if a Postman request requires file upload"""
    try:
        body = item["request"].get("body", {})
        mode = body.get("mode")
        
        if mode in ("file", "binary", "formdata"):
            if mode == "formdata":
                # Check if any formdata field is of type 'file'
                for entry in body.get("formdata", []):
                    if entry.get("type") == "file":
                        return True
            return mode in ("file", "binary")
        return False
    except KeyError:
        return False


def extract_items(collection: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all request items from Postman collection (handles nested folders)"""
    items = []
    
    def walk(node):
        if isinstance(node, dict) and "item" in node:
            for it in node["item"]:
                walk(it)
        elif isinstance(node, dict) and "request" in node:
            items.append(node)
    
    walk(collection)
    return items


def pick_random_file(doc_dir: str) -> Optional[str]:
    """Pick a random file from documents directory"""
    candidates = []
    for root, _, files in os.walk(doc_dir):
        for name in files:
            if name.startswith("."):
                continue
            candidates.append(os.path.join(root, name))
    
    return random.choice(candidates) if candidates else None


# ==================== GEMINI AI FUNCTIONS ====================

# Cache for uploaded files to avoid re-uploading
_uploaded_cache: Dict[str, Any] = {}


def gemini_upload(path: str):
    """Upload file to Gemini API (with caching)"""
    abspath = str(Path(path).resolve())
    if abspath in _uploaded_cache:
        return _uploaded_cache[abspath]
    
    # Let SDK infer mime type; supports images & PDFs
    file_obj = genai.upload_file(abspath)
    _uploaded_cache[abspath] = file_obj
    return file_obj


def gemini_json_from_prompt(
    system_prompt_path: str,
    user_payload: str,
    files: Optional[List[Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Call Gemini with a prompt and optional files, extract JSON from response
    """
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            sys_prompt = f.read()
        
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Build prompt parts
        parts = [sys_prompt, "\n\n```json_input\n", user_payload, "\n```"]
        inputs = []
        
        if files:
            inputs.extend(files)
        inputs.append("\n".join(parts))
        
        # Generate content
        resp = model.generate_content(inputs)
        text = (resp.text or "").strip()
        
        # Extract JSON from response
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        
        return json.loads(m.group(0))
    
    except Exception as e:
        log(f"Gemini call failed: {e}")
        return None


def classify_document_with_gemini(file_path: str) -> Optional[DocClassification]:
    """
    Use Gemini Vision to perform OCR and classify document type
    """
    try:
        file_ref = gemini_upload(file_path)
        data = gemini_json_from_prompt(
            "prompts/classify_document.md",
            user_payload="Classify the attached file. Return JSON only.",
            files=[file_ref],
        )
        
        if not data:
            return None
        
        return DocClassification(**data)
    
    except Exception as e:
        log(f"Document classification failed for {file_path}: {e}")
        return None


def normalize_error_with_gemini(
    status: int,
    headers: Dict[str, str],
    body: str
) -> Optional[ErrorStruct]:
    """
    Use Gemini to normalize vague error responses into standard format
    """
    payload = json.dumps({
        "status": status,
        "headers": headers,
        "body": body
    })
    
    data = gemini_json_from_prompt("prompts/normalize_error.md", payload)
    if not data:
        return None
    
    try:
        return ErrorStruct(**data)
    except Exception:
        return None


# ==================== QUICK ERROR PARSING (BEFORE LLM) ====================

EXT_HINTS = [".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".tiff"]
DOC_HINTS = [
    ("pan", "PAN card"),
    ("aadhaar", "Aadhaar card"),
    ("aadhar", "Aadhaar card"),
    ("passport", "passport"),
    ("driving", "driving_license"),
    ("license", "driving_license"),
    ("utility", "utility_bill"),
    ("bank", "bank_statement"),
]


def cheap_error_to_struct(body_text: str) -> Optional[ErrorStruct]:
    """
    Quick pattern matching for common error formats (before calling LLM)
    """
    lower = body_text.lower()
    
    # Look for extension hints
    ext = None
    for e in EXT_HINTS:
        if e in lower:
            ext = e
            break
    
    # Look for document type hints
    doc = None
    for key, val in DOC_HINTS:
        if key in lower:
            doc = val
            break
    
    if ext or doc:
        return ErrorStruct(
            required_extension_type=ext,
            required_document_type=doc,
            description=body_text.strip()[:800]
        )
    
    # Check for content-type mentions
    m = re.search(r"(content[- ]type|mime):\s*([\w/+.\-]+)", lower)
    if m:
        return ErrorStruct(
            required_extension_type=None,
            required_document_type=None,
            description=body_text.strip()[:800]
        )
    
    return None


# ==================== POSTMAN REQUEST BUILDER ====================

def build_request(
    item: Dict[str, Any],
    env: Dict[str, str],
    file_path: Optional[str]
) -> Tuple[str, str, Dict[str, str], Dict[str, Any]]:
    """
    Build HTTP request from Postman item
    Returns: (method, url, headers, payload_dict)
    """
    req = item["request"]
    method = req.get("method", "POST").upper()
    
    # Parse URL
    url = req.get("url")
    raw = url.get("raw", "") if isinstance(url, dict) else str(url)
    raw = resolve_postman_vars(raw, env)
    
    # Parse headers
    headers = {}
    for h in req.get("header", []):
        if not h.get("key"):
            continue
        headers[h["key"]] = resolve_postman_vars(h.get("value", ""), env)
    
    # Parse body
    body = req.get("body", {})
    mode = body.get("mode")
    
    data = {}
    files = None
    content = None
    
    if mode == "formdata":
        fields = []
        for f in body.get("formdata", []):
            key = f.get("key")
            typ = f.get("type")
            
            if typ == "file" and file_path:
                # Open file and read content
                with open(file_path, "rb") as fp:
                    file_content = fp.read()
                
                # Determine MIME type based on extension
                import mimetypes
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    # Default MIME types for common formats
                    ext = os.path.splitext(file_path)[1].lower()
                    mime_map = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.pdf': 'application/pdf',
                        '.doc': 'application/msword',
                        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    }
                    mime_type = mime_map.get(ext, 'application/octet-stream')
                
                # Send as tuple: (filename, file_content, mime_type)
                fields.append((key, (os.path.basename(file_path), file_content, mime_type)))
            elif typ == "text":
                fields.append((key, resolve_postman_vars(f.get("value", ""), env)))
        
        files = fields
    
    elif mode in ("file", "binary") and file_path:
        content = open(file_path, "rb").read()
    
    else:
        if body.get("raw"):
            content = resolve_postman_vars(body["raw"], env).encode("utf-8")
    
    return method, raw, headers, {"files": files, "data": data, "content": content}


# ==================== MAIN RUNNER ====================

def run(
    postman_path: str,
    env_path: Optional[str],
    docs_dir: str,
    out_dir: str,
    random_file_per_api: bool
):
    """
    Main execution function:
    1. Pre-classify ALL documents and create a map
    2. Load Postman collection
    3. For each upload API:
       - Pick a random document
       - Upload to API
       - Parse error response to understand requirements
       - Pick correct document from map
       - Retry with correct document
    4. Generate reports
    """
    # Setup
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    
    # Clean up old report if exists
    report_file = os.path.join(out_dir, "report.json")
    if os.path.exists(report_file):
        os.remove(report_file)
        log("🗑️  Removed old report.json")
    
    # === STEP 1: Pre-classify ALL documents ===
    log("Pre-classifying documents...")
    
    doc_map: Dict[str, DocClassification] = {}  # filename -> classification
    doc_map_file = os.path.join(out_dir, "document_classifications.json")
    
    # Check if we have a cached classification map
    if os.path.exists(doc_map_file):
        log(f"Loading cached classifications from {doc_map_file}")
        try:
            with open(doc_map_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
                for path, cls_dict in cached.items():
                    if os.path.exists(path):  # Only load if file still exists
                        doc_map[path] = DocClassification(**cls_dict)
            log(f"Loaded {len(doc_map)} cached classifications")
        except Exception as e:
            log(f"Failed to load cache: {e}")
            doc_map = {}
    
    # Find all documents
    doc_files = []
    for root, _, files in os.walk(docs_dir):
        for name in files:
            if name.startswith(".") or name.endswith(".md"):
                continue
            doc_files.append(os.path.join(root, name))
    
    # Classify only new/unclassified documents
    new_docs = [f for f in doc_files if f not in doc_map]
    
    if new_docs:
        log(f"Classifying {len(new_docs)} new documents...")
        for file_path in new_docs:
            log(f"  {os.path.basename(file_path)}", end=" → ")
            doc_cls = classify_document_with_gemini(file_path)
            if doc_cls:
                doc_map[file_path] = doc_cls
                log(f"{doc_cls.document_type} ({doc_cls.confidence:.2f})")
            else:
                log(f"classification failed")
                # Store failed classification
                doc_map[file_path] = DocClassification(document_type="classification_failed", confidence=0.0)
    else:
        log("All documents already classified")
    
    # Save updated classification map
    with open(doc_map_file, "w", encoding="utf-8") as f:
        serializable_map = {
            path: cls.model_dump() for path, cls in doc_map.items()
        }
        json.dump(serializable_map, f, indent=2, ensure_ascii=False)
    
    log(f"Total documents: {len(doc_map)}\n")
    
    # === STEP 2: Load Postman data ===
    log("Loading Postman collection...")
    env_vars = load_env_vars(env_path)
    coll = load_postman(postman_path)
    items = [it for it in extract_items(coll) if is_upload_request(it)]
    
    log(f"Found {len(items)} upload APIs\n")
    
    results: List[ApiResult] = []
    
    # === Track globally mapped documents across all APIs ===
    globally_mapped_docs = set()  # Files that have been successfully mapped to any API
    
    # === STEP 3: Test each API ===
    log("Testing APIs...\n")
    
    for item in items:
        name = item.get("name", "unnamed")
        req = item["request"]
        method = req.get("method", "POST").upper()
        
        log(f"API: {name}")
        
        # Pick random file from classified docs (exclude already mapped ones)
        if not doc_map:
            log("  No documents available")
            continue
        
        # Filter out already mapped documents
        available_docs = {path: cls for path, cls in doc_map.items() if path not in globally_mapped_docs}
        
        if not available_docs:
            log("  All documents already mapped")
            # Create empty result for this API
            result = ApiResult(
                api_name=name,
                method=method,
                url=f"{req.get('url', {}).get('raw', '') if isinstance(req.get('url'), dict) else req.get('url', '')}",
                picked_file=None,
                local_file_classification=None,
                error_struct=None,
                raw_response={"status": None, "message": "All documents already mapped"}
            )
            results.append(result)
            continue
        
        # Group documents by type for comprehensive testing
        docs_by_type = {}
        for path, cls in available_docs.items():
            doc_type = cls.document_type or "unknown"
            if doc_type not in docs_by_type:
                docs_by_type[doc_type] = []
            docs_by_type[doc_type].append((path, cls))
        
        # Start with a random document from available ones
        file_path = random.choice(list(available_docs.keys()))
        picked_name = file_path
        doc_cls = available_docs[file_path]
        
        log(f"  Testing: {os.path.basename(file_path)} ({doc_cls.document_type})")
        
        # === STEP 3A: First attempt with random document ===
        method_str, url, headers, payload = build_request(item, env_vars, file_path)
        
        status = None
        resp_text = ""
        resp_headers = {}
        
        try:
            if payload["files"] is not None:
                r = requests.request(method_str, url, headers=headers, files=payload["files"], timeout=60)
            elif payload["content"] is not None:
                r = requests.request(method_str, url, headers=headers, data=payload["content"], timeout=60)
            else:
                r = requests.request(method_str, url, headers=headers, data=payload["data"], timeout=60)
            
            status = r.status_code
            resp_headers = {k: v for k, v in r.headers.items()}
            
            try:
                resp_text = r.text
            except Exception:
                resp_text = "<non-text response>"
        
        except Exception as e:
            status = -1
            resp_text = f"<request failed: {e}>"
        
        # === STEP 3B: Parse error response (3-tier strategy) ===
        error_struct: Optional[ErrorStruct] = None
        
        # Tier 1: Check if API already returns our standard format
        try:
            j = json.loads(resp_text)
            if all(k in j for k in ["required_extension_type", "required_document_type", "description"]):
                error_struct = ErrorStruct(**j)
        except Exception:
            pass
        
        # Tier 2: Quick pattern matching
        if not error_struct:
            cheap = cheap_error_to_struct(resp_text)
            if cheap:
                error_struct = cheap
        
        # Tier 3: LLM fallback for vague errors
        if not error_struct and status not in [200, 201, 204]:
            error_struct = normalize_error_with_gemini(status or 0, resp_headers, resp_text or "")
        
        # === STEP 3C: Retry with correct document if needed ===
        retry_result = None
        
        # If API succeeded, no need to retry
        if status in [200, 201, 204]:
            log(f"  ✓ Success ({doc_cls.document_type})")
        
        # If API failed and we know what it wants, retry with correct doc
        elif error_struct and error_struct.required_document_type and error_struct.required_document_type != "unknown":
            required_type = error_struct.required_document_type
            
            # Check if we already sent the right document type
            if doc_cls.document_type and required_type.lower() in doc_cls.document_type.lower():
                pass  # Already correct type, no retry needed
            else:
                # Find matching document from map
                matching_docs = [
                    path for path, cls in doc_map.items()
                    if cls.document_type and required_type.lower() in cls.document_type.lower()
                ]
                
                if matching_docs:
                    correct_file = matching_docs[0]
                    correct_cls = doc_map[correct_file]
                    
                    # Retry request - rebuild from scratch
                    retry_method, retry_url, retry_headers, retry_payload = build_request(item, env_vars, correct_file)
                    try:
                        if retry_payload["files"] is not None:
                            retry_r = requests.request(retry_method, retry_url, headers=retry_headers, files=retry_payload["files"], timeout=60)
                        elif retry_payload["content"] is not None:
                            retry_r = requests.request(retry_method, retry_url, headers=retry_headers, data=retry_payload["content"], timeout=60)
                        else:
                            retry_r = requests.request(retry_method, retry_url, headers=retry_headers, data=retry_payload["data"], timeout=60)
                        
                        retry_status = retry_r.status_code
                        
                        if 200 <= retry_status < 300:
                            log(f"  ✓ Retry success ({correct_cls.document_type})")
                        
                        retry_result = {
                            "file": correct_file,
                            "classification": correct_cls,
                            "status": retry_status,
                            "body": retry_r.text[:1000]
                        }
                    
                    except Exception as e:
                        pass  # Retry failed
        
        # Store result
        result = ApiResult(
            api_name=name,
            method=method_str,
            url=url,
            picked_file=picked_name,
            local_file_classification=doc_cls,
            error_struct=error_struct,
            raw_response={
                "status": status,
                "headers": resp_headers,
                "body": resp_text[:4000],  # Truncate large responses
                "retry": retry_result
            }
        )
        results.append(result)
        
        tested_files = {picked_name}  # Already tested initial doc
        if retry_result:
            tested_files.add(retry_result["file"])
        
        all_test_results = []
        
        # Track which document types have been successfully accepted by this API
        accepted_doc_types = set()
        
        # If initial test succeeded, note the doc type
        if status in [200, 201, 204]:
            accepted_doc_types.add(doc_cls.document_type)
            globally_mapped_docs.add(picked_name)  # Mark as mapped
        
        # If retry succeeded, note that doc type
        if retry_result and 200 <= retry_result["status"] < 300:
            accepted_doc_types.add(retry_result["classification"].document_type)
            globally_mapped_docs.add(retry_result["file"])  # Mark as mapped
        
        # If we found what the API accepts, just auto-map remaining documents of that type
        if accepted_doc_types:
            
            for test_file, test_cls in doc_map.items():
                if test_file in tested_files:
                    continue  # Skip already tested files
                
                # Skip already mapped documents
                if test_file in globally_mapped_docs:
                    continue
                
                doc_type = test_cls.document_type or "unknown"
                
                # Skip if classification failed
                if doc_type == "classification_failed":
                    all_test_results.append({
                        "file": test_file,
                        "doc_type": "classification_failed",
                        "status": None,
                        "success": False,
                        "error": "Document classification failed",
                        "skipped": True
                    })
                    continue
                
                if doc_type == "unknown":
                    all_test_results.append({
                        "file": test_file,
                        "doc_type": "unknown",
                        "status": None,
                        "success": False,
                        "error": "Unknown document type",
                        "skipped": True
                    })
                    continue
                
                # If this doc type matches what API accepts, auto-map it
                if doc_type in accepted_doc_types:
                    globally_mapped_docs.add(test_file)  # Mark as mapped
                    all_test_results.append({
                        "file": test_file,
                        "doc_type": doc_type,
                        "status": 200,
                        "success": True,
                        "auto_mapped": True
                    })
                else:
                    # Different doc type - we know it won't work, skip API call
                    all_test_results.append({
                        "file": test_file,
                        "doc_type": doc_type,
                        "status": 400,
                        "success": False,
                        "error": f"API only accepts {', '.join(accepted_doc_types)}",
                        "auto_rejected": True
                    })
        else:
            # No successful response yet - test all remaining documents to find what works
            
            for test_file, test_cls in doc_map.items():
                if test_file in tested_files:
                    continue
                
                # Skip already mapped documents
                if test_file in globally_mapped_docs:
                    continue
                
                doc_type = test_cls.document_type or "unknown"
                
                if doc_type in ["classification_failed", "unknown"]:
                    all_test_results.append({
                        "file": test_file,
                        "doc_type": doc_type,
                        "status": None,
                        "success": False,
                        "error": f"{doc_type.replace('_', ' ').title()}",
                        "skipped": True
                    })
                    continue
                
                # Build and send request
                test_method, test_url, test_headers, test_payload = build_request(item, env_vars, test_file)
                
                try:
                    if test_payload["files"] is not None:
                        test_r = requests.request(test_method, test_url, headers=test_headers, files=test_payload["files"], timeout=60)
                    elif test_payload["content"] is not None:
                        test_r = requests.request(test_method, test_url, headers=test_headers, data=test_payload["content"], timeout=60)
                    else:
                        test_r = requests.request(test_method, test_url, headers=test_headers, data=test_payload["data"], timeout=60)
                    
                    test_status = test_r.status_code
                    
                    if 200 <= test_status < 300:
                        accepted_doc_types.add(doc_type)
                        globally_mapped_docs.add(test_file)  # Mark as mapped
                        all_test_results.append({
                            "file": test_file,
                            "doc_type": doc_type,
                            "status": test_status,
                            "success": True
                        })
                        # Break out and auto-map remaining docs of this type
                        break
                    else:
                        all_test_results.append({
                            "file": test_file,
                            "doc_type": doc_type,
                            "status": test_status,
                            "success": False,
                            "error": test_r.text[:200]
                        })
                except Exception as e:
                    all_test_results.append({
                        "file": test_file,
                        "doc_type": doc_type,
                        "status": None,
                        "success": False,
                        "error": str(e)
                    })
        
        # Add comprehensive test results to the result
        result.raw_response["all_document_tests"] = all_test_results
    
    # Generate simplified report
    simplified_report = []
    for r in results:
        # Collect all successful files
        correct_files = []
        failed_files = []
        skipped_files = []
        
        # Check if initial attempt succeeded
        if r.raw_response.get("status") in [200, 201, 204]:
            correct_files.append({
                "fileName": os.path.basename(r.picked_file),
                "docType": r.local_file_classification.document_type if r.local_file_classification else "unknown"
            })
        else:
            failed_files.append({
                "nameOfFile": os.path.basename(r.picked_file),
                "docType": r.local_file_classification.document_type if r.local_file_classification else "unknown",
                "errorMessage": r.raw_response.get("body", "")[:200]
            })
        
        # Check if retry succeeded
        if r.raw_response.get("retry"):
            if r.raw_response["retry"]["status"] in [200, 201, 204]:
                correct_files.append({
                    "fileName": os.path.basename(r.raw_response["retry"]["file"]),
                    "docType": r.raw_response["retry"]["classification"].document_type
                })
            else:
                failed_files.append({
                    "nameOfFile": os.path.basename(r.raw_response["retry"]["file"]),
                    "docType": r.raw_response["retry"]["classification"].document_type,
                    "errorMessage": r.raw_response["retry"].get("body", "")[:200]
                })
        
        # Check all comprehensive test results
        if r.raw_response.get("all_document_tests"):
            for test in r.raw_response["all_document_tests"]:
                if test.get("skipped"):
                    # Classification failed or unknown
                    skipped_files.append({
                        "fileName": os.path.basename(test["file"]),
                        "reason": test.get("error", "Unknown reason")
                    })
                elif test.get("success"):
                    correct_files.append({
                        "fileName": os.path.basename(test["file"]),
                        "docType": test["doc_type"],
                        "autoMapped": test.get("auto_mapped", False)
                    })
                elif test.get("auto_rejected"):
                    # Document type doesn't match what API accepts - auto-rejected
                    failed_files.append({
                        "nameOfFile": os.path.basename(test["file"]),
                        "docType": test["doc_type"],
                        "errorMessage": test.get("error", ""),
                        "autoRejected": True
                    })
                else:
                    failed_files.append({
                        "nameOfFile": os.path.basename(test["file"]),
                        "docType": test["doc_type"],
                        "errorMessage": test.get("error", "")[:200]
                    })
        
        api_report = {
            "api_name": r.api_name,
            "path": r.url,
            "accepted_documents": correct_files,
            "rejected_documents": failed_files
        }
        
        # Only add skipped_files if there are any
        if skipped_files:
            api_report["skipped_documents"] = skipped_files
        
        simplified_report.append(api_report)
    
    # Save report
    report_file = os.path.join(out_dir, "report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(simplified_report, f, indent=2, ensure_ascii=False)
    
    log(f"Report saved: {report_file}")


# ==================== CLI ENTRY POINT ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="API Document Mapper - Test file upload APIs and map document requirements"
    )
    parser.add_argument(
        "--postman",
        required=True,
        help="Path to Postman collection JSON"
    )
    parser.add_argument(
        "--env",
        help="Path to Postman environment JSON (optional)"
    )
    parser.add_argument(
        "--docs",
        required=True,
        help="Directory containing test documents"
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory for reports (default: outputs)"
    )
    parser.add_argument(
        "--random-file-per-api",
        action="store_true",
        help="Pick a random file for each API (recommended)"
    )
    
    args = parser.parse_args()
    
    run(
        postman_path=args.postman,
        env_path=args.env,
        docs_dir=args.docs,
        out_dir=args.out,
        random_file_per_api=args.random_file_per_api,
    )
