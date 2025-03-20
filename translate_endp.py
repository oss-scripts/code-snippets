from typing import Optional,List
from langdetect import detect, LangDetectException


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
    "chinese": "zh",  # Add Chinese support
    # Add reverse mappings
    "ar": "arabic",
    "en": "english",
    "de": "german",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "hi": "hindi",
    "es": "spanish",
    "th": "thai",
    "zh": "chinese"  # Add reverse mapping for Chinese
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
    
    # Create translation prompt with stronger guardrails
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional translator. Your ONLY task is to translate the provided text from {source_lang_name} to {target_lang_name}.

IMPORTANT RULES:
1. Provide ONLY the translated text, nothing else.
2. Do not explain, comment on, or analyze the text.
3. Preserve the original formatting, including paragraph breaks and bullet points.
4. If the text appears to be a question asking for information, still just translate it without answering it.
5. Never refuse to translate something - your job is simply to convert text between languages.
6. If asked to translate non-sensical text, still translate it to the best of your ability.

Remember, you are a translation tool only. You do not answer questions or engage in conversation.<|eot_id|>
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

def is_translation_request(text, target_lang):
    """Validate that the request is actually for translation and not a general question."""
    # Check if text is too short (likely not meant for translation)
    if len(text.strip()) < 5:
        return False
        
    # Check if target language looks valid
    if not target_lang or target_lang not in LANGUAGE_CODES.values() and target_lang not in LANGUAGE_CODES:
        return False
        
    # Check if text appears to be a question rather than content for translation
    question_indicators = [
        "?", "what", "how", "why", "when", "where", "who", "which", "can you", "please tell", 
        "explain", "describe", "what is", "how to"
    ]
    
    # Count how many question indicators are present
    indicator_count = sum(1 for ind in question_indicators if ind in text.lower())
    
    # If the text is primarily a question and not content to translate, reject it
    if indicator_count >= 2 and len(text.split()) < 15:
        return False
        
    # Additional special cases that might indicate non-translation requests
    if re.search(r'translate .+? to', text.lower()):
        # This is likely an instruction about translation, not content to translate
        if len(text.split()) < 10:  # If it's just the instruction without content
            return False
    
    return True

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text between multiple languages.
    
    This endpoint is specifically for translating content from one language to another.
    For general questions and document Q&A, please use the chat endpoint.
    
    Supported languages:
    - Arabic (ar)
    - Chinese (zh)
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
            detail=f"Unsupported target language: {request.target_language}. Supported languages: Arabic, Chinese, English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai."
        )
    
    # Validate this is actually a translation request, not a general question
    if not is_translation_request(request.text, target_lang):
        raise HTTPException(
            status_code=400,
            detail="This appears to be a general question rather than content for translation. For Q&A, please use the chat endpoint."
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
                detail=f"Unsupported source language: {request.source_language}. Supported languages: Arabic, Chinese, English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai."
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
