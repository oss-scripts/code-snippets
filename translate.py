import os
import sys
import argparse
import requests
from tqdm import tqdm
import time
import json
from langchain.document_loaders import PDFPlumberLoader

def translate_text_with_llm(arabic_text, batch_size=1500):
    """Translate Arabic text to English using the existing VLLM endpoint"""
    # Break text into manageable chunks to avoid context length issues
    text_chunks = []
    for i in range(0, len(arabic_text), batch_size):
        text_chunks.append(arabic_text[i:i+batch_size])
    
    all_translated = []
    
    for chunk in tqdm(text_chunks, desc="Translating text chunks"):
        # Create translation prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional translator from Arabic to English. Translate the following Arabic text to English.
Preserve the original formatting as much as possible, including paragraph breaks.
Do not add any explanations or notes - just return the fluent English translation.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Translate this Arabic text to English:

{chunk}<|eot_id|>
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
                all_translated.append(translated_text.strip())
            else:
                print(f"Error from API: {response.status_code}")
                all_translated.append(f"[Translation error: API returned {response.status_code}]")
                
        except Exception as e:
            print(f"Error during translation: {str(e)}")
            all_translated.append(f"[Translation error: {str(e)}]")
        
        # Add a small delay between requests to avoid overwhelming the API
        time.sleep(0.5)
    
    return " ".join(all_translated)

def translate_specific_page(input_pdf_path, output_txt_path, page_number):
    """Load PDF, translate specific page content, and save to text file"""
    print(f"Loading PDF: {input_pdf_path}")
    
    # Use PDFPlumberLoader from your existing langchain setup
    loader = PDFPlumberLoader(input_pdf_path)
    pages = loader.load()
    
    total_pages = len(pages)
    print(f"PDF has {total_pages} pages in total")
    
    # Verify page number is valid (convert from 1-indexed to 0-indexed)
    page_index = page_number - 1
    if page_index < 0 or page_index >= total_pages:
        print(f"Error: Page {page_number} does not exist. PDF has {total_pages} pages.")
        return False
    
    # Get the specified page
    page = pages[page_index]
    page_text = page.page_content
    
    # Open output text file
    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        outfile.write(f"================ PAGE {page_number} ================\n\n")
        
        if not page_text.strip():
            print(f"Page {page_number} appears to be empty or contains only images")
            outfile.write("[No text content or contains only images]\n")
        else:
            print(f"Translating page {page_number}")
            translated_text = translate_text_with_llm(page_text)
            
            # Write translated text to file
            outfile.write(f"{translated_text}\n")
    
    print(f"Translation complete! Saved to: {output_txt_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Translate specific page from Arabic PDF to English text file')
    parser.add_argument('input_pdf', help='Path to the input Arabic PDF')
    parser.add_argument('--page', type=int, required=True, help='Page number to translate (starting from 1)')
    parser.add_argument('--output_txt', help='Path for the output text file (default: input_name_page{N}_translated.txt)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_pdf):
        print(f"Error: Input PDF file not found: {args.input_pdf}")
        sys.exit(1)
    
    if args.output_txt:
        output_txt = args.output_txt
    else:
        base_name = os.path.splitext(args.input_pdf)[0]
        output_txt = f"{base_name}_page{args.page}_translated.txt"
    
    translate_specific_page(args.input_pdf, output_txt, args.page)

if __name__ == "__main__":
    main()
