You will receive an API error response (status, headers, body). Extract a uniform JSON object that describes what the API expects.

**Return JSON only in this exact schema:**
```json
{
  "required_extension_type": ".pdf | .jpg | .png | .doc | .docx | .jpeg | .tiff | unknown",
  "required_document_type": "PAN card | Aadhaar card | passport | driving_license | bank_statement | utility_bill | unknown",
  "description": "human-friendly explanation of the error and what to upload"
}
```

**Rules:**
1. Prefer the most specific inference supported by the evidence
2. If multiple extensions or docs are acceptable and you cannot choose, return "unknown"
3. Be concise. Do not invent requirements that aren't supported by the error message
4. Extract file format hints from error messages (e.g., "PDF required", "invalid image format")
5. Extract document type hints from error messages (e.g., "PAN card needed", "identity proof required")
6. Output JSON only, no additional text
