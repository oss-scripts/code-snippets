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

def translate_pdf_to_txt(input_pdf_path, output_txt_path):
    """Load PDF, translate content, and save to text file"""
    print(f"Loading PDF: {input_pdf_path}")
    
    # Use PDFPlumberLoader from your existing langchain setup
    loader = PDFPlumberLoader(input_pdf_path)
    pages = loader.load()
    
    total_pages = len(pages)
    print(f"Total pages: {total_pages}")
    
    # Open output text file
    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        # Process each page
        for i, page in enumerate(tqdm(pages, desc="Translating pages")):
            page_text = page.page_content
            
            # Write page header
            outfile.write(f"================ PAGE {i+1} ================\n\n")
            
            if not page_text.strip():
                print(f"Page {i+1} appears to be empty or contains only images")
                outfile.write("[No text content or contains only images]\n\n")
                continue
                
            print(f"Translating page {i+1}/{total_pages}")
            translated_text = translate_text_with_llm(page_text)
            
            # Write translated text to file
            outfile.write(f"{translated_text}\n\n")
            
            # Add page separator
            if i < total_pages - 1:
                outfile.write("\n" + "="*50 + "\n\n")
    
    print(f"Translation complete! Saved to: {output_txt_path}")

def main():
    parser = argparse.ArgumentParser(description='Translate Arabic PDF to English text file')
    parser.add_argument('input_pdf', help='Path to the input Arabic PDF')
    parser.add_argument('--output_txt', help='Path for the output text file (default: input_name_translated.txt)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_pdf):
        print(f"Error: Input PDF file not found: {args.input_pdf}")
        sys.exit(1)
    
    if args.output_txt:
        output_txt = args.output_txt
    else:
        base_name = os.path.splitext(args.input_pdf)[0]
        output_txt = f"{base_name}_translated.txt"
    
    translate_pdf_to_txt(args.input_pdf, output_txt)

if __name__ == "__main__":
    main()
