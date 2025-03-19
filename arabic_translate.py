import os
import sys
import argparse
import requests
import time
from tqdm import tqdm
import PyPDF2  # Usually pre-installed or easy to install

def translate_text_with_llm(arabic_text, batch_size=1500):
    """Translate Arabic text to English using the existing VLLM endpoint"""
    if not arabic_text or arabic_text.isspace():
        return "[No extractable text content]"
        
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
If there are unreadable characters or OCR artifacts, try to make sense of the text where possible.
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

def extract_text_from_pdf(pdf_path, page_number=None):
    """Extract text from PDF, with option to extract specific page only"""
    try:
        # Open the PDF
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # If page number specified, extract only that page
            if page_number is not None:
                # Convert from 1-indexed to 0-indexed
                page_index = page_number - 1
                if page_index < 0 or page_index >= total_pages:
                    return None, f"Error: Page {page_number} does not exist. PDF has {total_pages} pages."
                
                # Extract single page
                page = pdf_reader.pages[page_index]
                text = page.extract_text()
                return {page_number: text}, total_pages
            
            # Extract all pages
            else:
                pages_text = {}
                for i in range(total_pages):
                    page = pdf_reader.pages[i]
                    pages_text[i+1] = page.extract_text()
                return pages_text, total_pages
                
    except Exception as e:
        return None, f"Error extracting text: {str(e)}"

def translate_specific_page(input_pdf_path, output_txt_path, page_number):
    """Extract specific page from PDF, translate it, and save to text file"""
    print(f"Loading PDF: {input_pdf_path}")
    
    # Extract text from the specified page
    page_text, page_info = extract_text_from_pdf(input_pdf_path, page_number)
    
    if not page_text:
        print(page_info)  # Print error message
        return False
    
    print(f"PDF has {page_info} pages in total")
    
    # Open output text file
    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        outfile.write(f"================ PAGE {page_number} ================\n\n")
        
        extracted_text = page_text[page_number]
        if not extracted_text or extracted_text.isspace():
            print(f"Page {page_number} appears to be empty, contains images only, or text couldn't be extracted")
            outfile.write("[No extractable text content or contains only images]\n")
        else:
            print(f"Translating page {page_number}")
            print(f"Extracted text sample: {extracted_text[:100]}...")  # Debug: show sample of extracted text
            
            translated_text = translate_text_with_llm(extracted_text)
            
            # Write translated text to file
            outfile.write(f"{translated_text}\n")
    
    print(f"Translation complete! Saved to: {output_txt_path}")
    return True

def translate_all_pages(input_pdf_path, output_txt_path):
    """Extract all pages from PDF, translate them, and save to text file"""
    print(f"Loading PDF: {input_pdf_path}")
    
    # Extract text from all pages
    pages_text, total_pages = extract_text_from_pdf(input_pdf_path)
    
    if not pages_text:
        print(total_pages)  # Print error message
        return False
    
    print(f"PDF has {total_pages} pages in total")
    
    # Open output text file
    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        # Process each page
        for page_number in tqdm(range(1, total_pages + 1), desc="Processing pages"):
            extracted_text = pages_text.get(page_number, "")
            
            # Write page header
            outfile.write(f"================ PAGE {page_number} ================\n\n")
            
            if not extracted_text or extracted_text.isspace():
                print(f"Page {page_number} appears to be empty, contains images only, or text couldn't be extracted")
                outfile.write("[No extractable text content or contains only images]\n\n")
                continue
                
            print(f"Translating page {page_number}/{total_pages}")
            translated_text = translate_text_with_llm(extracted_text)
            
            # Write translated text to file
            outfile.write(f"{translated_text}\n\n")
            
            # Add page separator
            if page_number < total_pages:
                outfile.write("\n" + "="*50 + "\n\n")
    
    print(f"Translation complete! Saved to: {output_txt_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Translate Arabic PDF to English text file')
    parser.add_argument('input_pdf', help='Path to the input Arabic PDF')
    parser.add_argument('--page', type=int, help='Specific page number to translate (starting from 1)')
    parser.add_argument('--output_txt', help='Path for the output text file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_pdf):
        print(f"Error: Input PDF file not found: {args.input_pdf}")
        sys.exit(1)
    
    # Determine output filename
    if args.output_txt:
        output_txt = args.output_txt
    else:
        base_name = os.path.splitext(args.input_pdf)[0]
        if args.page:
            output_txt = f"{base_name}_page{args.page}_translated.txt"
        else:
            output_txt = f"{base_name}_translated.txt"
    
    # Process either specific page or entire document
    if args.page:
        translate_specific_page(args.input_pdf, output_txt, args.page)
    else:
        translate_all_pages(args.input_pdf, output_txt)

if __name__ == "__main__":
    main()
