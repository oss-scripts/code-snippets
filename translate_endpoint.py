from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import requests
import time
from langdetect import detect, LangDetectException

# Import necessary components from your existing setup
from app2 import app  # Reuse your existing FastAPI app

class TranslationRequest(BaseModel):
    text: str
    target_language: str
    source_language: Optional[str] = None  # Optional: auto-detect if not provided

class TranslationResponse(BaseModel):
    translated_text: str
    detected_source_language: str
    target_language: str

# Language code mapping
LANGUAGE_CODES = {
    "arabic": "ar",
    "english": "en",
    "german": "de",
    "french": "fr", 
    "italian": "it",
    "portuguese": "pt",
    "hindi": "hi",
    "spanish": "es",
    "thai": "th",
    # Add reverse mappings
    "ar": "arabic",
    "en": "english",
    "de": "german",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "hi": "hindi",
    "es": "spanish",
    "th": "thai"
}

def detect_language(text):
    """Detect the language of the input text."""
    try:
        lang_code = detect(text)
        if lang_code in LANGUAGE_CODES:
            return lang_code
        else:
            # Default to English if detected language is not supported
            return "en"
    except LangDetectException:
        # Default to English if detection fails
        return "en"

def translate_with_llm(text, target_lang, source_lang=None):
    """Translate text using the existing VLLM endpoint."""
    if not source_lang:
        source_lang = detect_language(text)
    
    # Get full language names for the prompt
    source_lang_name = LANGUAGE_CODES.get(source_lang, source_lang)
    target_lang_name = LANGUAGE_CODES.get(target_lang, target_lang)
    
    # Create translation prompt
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional translator. Translate the following text from {source_lang_name} to {target_lang_name}.
Preserve the original formatting, including paragraph breaks and bullet points.
Do not add any explanations or notes - just return the translated text.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Translate this text from {source_lang_name} to {target_lang_name}:

{text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    # Call the VLLM API using your existing endpoint
    try:
        response = requests.post(
            "http://localhost:8503/llama_generate",
            json={
                "prompt": [prompt],
                "kwargs": {
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "top_p": 0.9
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            translated_text = result[0]['outputs'][0]['text']
            return translated_text.strip(), source_lang
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Translation API error: {response.status_code}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation error: {str(e)}"
        )

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text between multiple languages.
    
    Supported languages:
    - Arabic (ar)
    - English (en)
    - German (de)
    - French (fr)
    - Italian (it)
    - Portuguese (pt)
    - Hindi (hi)
    - Spanish (es)
    - Thai (th)
    
    You can specify the language code or the full language name.
    If source language is not provided, it will be auto-detected.
    """
    # Validate and normalize target language
    target_lang = request.target_language.lower()
    if target_lang in LANGUAGE_CODES:
        target_lang = LANGUAGE_CODES[target_lang]
    elif target_lang not in LANGUAGE_CODES.values():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language: {request.target_language}. Supported languages: Arabic, English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai."
        )
    
    # Process source language if provided
    source_lang = None
    if request.source_language:
        source_lang = request.source_language.lower()
        if source_lang in LANGUAGE_CODES:
            source_lang = LANGUAGE_CODES[source_lang]
        elif source_lang not in LANGUAGE_CODES.values():
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source language: {request.source_language}. Supported languages: Arabic, English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai."
            )
    
    # Translate text
    translated_text, detected_source = translate_with_llm(
        request.text, 
        target_lang,
        source_lang
    )
    
    # Return response
    return TranslationResponse(
        translated_text=translated_text,
        detected_source_language=LANGUAGE_CODES.get(detected_source, detected_source),
        target_language=LANGUAGE_CODES.get(target_lang, target_lang)
    )
