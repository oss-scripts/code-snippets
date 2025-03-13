from vllm import LLM, SamplingParams
import time

def translate_arabic_to_english(arabic_text, model_name="meta-llama/Llama-2-7b-chat-hf"):
    """
    Translates Arabic text to English using vLLM's Python API
    
    Args:
        arabic_text (str): Arabic text to translate
        model_name (str): Name of the model to use
        
    Returns:
        str: English translation
    """
    try:
        # Initialize the LLM
        # Note: This will load the model, which may take time on first execution
        llm = LLM(model=model_name)
        
        # Prepare the prompt with clear instructions
        prompt = f"""Translate the following Arabic text to English:

Arabic: {arabic_text}

English translation:"""
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,        # Lower temperature for more deterministic translations
            max_tokens=1000,        # Maximum length of generated text
            top_p=0.95,             # Nucleus sampling
            stop_token_ids=None,    # No special stop tokens
            stop=["Arabic:", "\n\n"] # Stop generation at these strings
        )
        
        # Generate the translation
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        end_time = time.time()
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text.strip()
        
        print(f"Translation completed in {end_time - start_time:.2f} seconds")
        
        return generated_text
    
    except Exception as e:
        print(f"Error generating translation: {e}")
        return None

# Example usage
if __name__ == "__main__":
    arabic_text = "مرحبا بالعالم. كيف حالك اليوم؟"  # "Hello world. How are you today?"
    translation = translate_arabic_to_english(arabic_text)
    
    if translation:
        print(f"Arabic: {arabic_text}")
        print(f"English: {translation}")
    else:
        print("Translation failed.")
