import os
import sys
import argparse
import requests
import time
from tqdm import tqdm
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

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

def create_pdf_with_translated_text(translated_pages, output_pdf_path, page_to_translate=None):
    """Create a PDF with translated text"""
    # Create a PDF document
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        rightMargin=72, 
        leftMargin=72,
        topMargin=72, 
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Handle single page or multiple pages
    if page_to_translate:
        # Add page header
        story.append(Paragraph(f"<b>Translated Page {page_to_translate}</b>", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        # Format text with proper paragraph breaks
        text = translated_pages[page_to_translate]
        paragraphs = text.split('\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
    else:
        # Process all pages
        for page_num in sorted(translated_pages.keys()):
            # Add page header
            story.append(Paragraph(f"<b>Page {page_num}</b>", styles['Heading1']))
            story.append(Spacer(1, 0.2*inch))
            
            # Format text with proper paragraph breaks
            text = translated_pages[page_num]
            paragraphs = text.split('\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para, styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            
            # Add page break between pages
            if page_num < max(translated_pages.keys()):
                story.append(Paragraph(" ", styles['Normal']))
                story.append(Spacer(1, 0.5*inch))
                story.append(Paragraph("<hr/>", styles['Normal']))
                story.append(Spacer(1, 0.5*inch))
    
    # Build the PDF
    doc.build(story)
    return True

def translate_specific_page(input_pdf_path, output_pdf_path, page_number):
    """Extract specific page from PDF, translate it, and save to PDF file"""
    print(f"Loading PDF: {input_pdf_path}")
    
    # Extract text from the specified page
    page_text, page_info = extract_text_from_pdf(input_pdf_path, page_number)
    
    if not page_text:
        print(page_info)  # Print error message
        return False
    
    print(f"PDF has {page_info} pages in total")
    
    # Dictionary to store translated text
    translated_pages = {}
    
    extracted_text = page_text[page_number]
    if not extracted_text or extracted_text.isspace():
        print(f"Page {page_number} appears to be empty, contains images only, or text couldn't be extracted")
        translated_pages[page_number] = "[No extractable text content or contains only images]"
    else:
        print(f"Translating page {page_number}")
        print(f"Extracted text sample: {extracted_text[:100]}...")  # Debug: show sample of extracted text
        
        translated_text = translate_text_with_llm(extracted_text)
        translated_pages[page_number] = translated_text
    
    # Create PDF with translated text
    create_pdf_with_translated_text(translated_pages, output_pdf_path, page_number)
    
    print(f"Translation complete! Saved to: {output_pdf_path}")
    return True

def translate_all_pages(input_pdf_path, output_pdf_path):
    """Extract all pages from PDF, translate them, and save to PDF file"""
    print(f"Loading PDF: {input_pdf_path}")
    
    # Extract text from all pages
    pages_text, total_pages = extract_text_from_pdf(input_pdf_path)
    
    if not pages_text:
        print(total_pages)  # Print error message
        return False
    
    print(f"PDF has {total_pages} pages in total")
    
    # Dictionary to store translated text for all pages
    translated_pages = {}
    
    # Process each page
    for page_number in tqdm(range(1, total_pages + 1), desc="Processing pages"):
        extracted_text = pages_text.get(page_number, "")
        
        if not extracted_text or extracted_text.isspace():
            print(f"Page {page_number} appears to be empty, contains images only, or text couldn't be extracted")
            translated_pages[page_number] = "[No extractable text content or contains only images]"
            continue
            
        print(f"Translating page {page_number}/{total_pages}")
        translated_text = translate_text_with_llm(extracted_text)
        translated_pages[page_number] = translated_text
    
    # Create PDF with all translated pages
    create_pdf_with_translated_text(translated_pages, output_pdf_path)
    
    print(f"Translation complete! Saved to: {output_pdf_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Translate Arabic PDF to English PDF file')
    parser.add_argument('input_pdf', help='Path to the input Arabic PDF')
    parser.add_argument('--page', type=int, help='Specific page number to translate (starting from 1)')
    parser.add_argument('--output_pdf', help='Path for the output PDF file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_pdf):
        print(f"Error: Input PDF file not found: {args.input_pdf}")
        sys.exit(1)
    
    # Determine output filename
    if args.output_pdf:
        output_pdf = args.output_pdf
    else:
        base_name = os.path.splitext(args.input_pdf)[0]
        if args.page:
            output_pdf = f"{base_name}_page{args.page}_translated.pdf"
        else:
            output_pdf = f"{base_name}_translated.pdf"
    
    # Process either specific page or entire document
    if args.page:
        translate_specific_page(args.input_pdf, output_pdf, args.page)
    else:
        translate_all_pages(args.input_pdf, output_pdf)

if __name__ == "__main__":
    main()
