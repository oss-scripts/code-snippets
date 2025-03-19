import os
import sys
import argparse
import requests
import time
import re
from tqdm import tqdm
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib import colors
from reportlab.lib.units import inch

def translate_text_with_llm(arabic_text, batch_size=1500, preserve_structure=True):
    """Translate Arabic text to English using the existing VLLM endpoint"""
    if not arabic_text or arabic_text.isspace():
        return "[No extractable text content]"
    
    # Structure preservation instructions
    structure_instructions = """
Pay careful attention to the document structure:
1. Preserve ALL headings and titles with their exact formatting (e.g., if a line is centered and appears to be a title, translate it as a title)
2. Keep all bullet points and numbered lists intact
3. Preserve paragraph breaks exactly as in the original
4. Maintain table-like structures where present
5. Keep any section numbering (1.1, 1.2, etc.)
6. If text appears to be in a special format (headers, footers, captions), maintain that distinction
7. For text that appears to be a heading (short, possibly centered or bold), translate it and indicate it's a heading by adding [HEADING] at the beginning
"""

    # Break text into manageable chunks to avoid context length issues
    text_chunks = []
    for i in range(0, len(arabic_text), batch_size):
        text_chunks.append(arabic_text[i:i+batch_size])
    
    all_translated = []
    
    for chunk in tqdm(text_chunks, desc="Translating text chunks"):
        # Create translation prompt with structure preservation
        structure_part = structure_instructions if preserve_structure else ""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional translator from Arabic to English. Translate the following Arabic text to English.
{structure_part}
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
                    text = page.extract_text()
                    pages_text[i+1] = text
                return pages_text, total_pages
                
    except Exception as e:
        return None, f"Error extracting text: {str(e)}"

def create_enhanced_pdf_with_translated_text(translated_pages, output_pdf_path, page_to_translate=None):
    """Create a PDF with translated text that preserves document structure"""
    # Create a PDF document
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        rightMargin=72, 
        leftMargin=72,
        topMargin=72, 
        bottomMargin=72
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Create custom styles for various elements
    title_style = ParagraphStyle(
        'Title', 
        parent=styles['Title'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        'Heading', 
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=10
    )
    
    subheading_style = ParagraphStyle(
        'SubHeading', 
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=8
    )
    
    normal_style = styles['Normal']
    
    story = []
    
    # Handle single page or multiple pages
    pages_to_process = [page_to_translate] if page_to_translate else sorted(translated_pages.keys())
    
    for page_num in pages_to_process:
        # Add page indicator
        story.append(Paragraph(f"<i>--- Page {page_num} ---</i>", 
                              ParagraphStyle('PageIndicator', alignment=TA_CENTER, textColor=colors.gray)))
        story.append(Spacer(1, 0.2*inch))
        
        # Process text with structure preservation
        text = translated_pages[page_num]
        if not text or text == "[No extractable text content or contains only images]":
            story.append(Paragraph(text, normal_style))
            if page_num != pages_to_process[-1]:  # Not the last page
                story.append(PageBreak())
            continue
            
        # Split text into lines and analyze structure
        lines = text.split('\n')
        
        # Process each line to identify structure
        for line in lines:
            line = line.strip()
            if not line:
                # Empty line - add space
                story.append(Spacer(1, 0.1*inch))
                continue
                
            # Check for headings marked by our translation instructions
            if line.startswith("[HEADING]"):
                line = line.replace("[HEADING]", "").strip()
                story.append(Paragraph(line, heading_style))
                continue
                
            # Check for likely headings (short lines that end with colon or short capitalized lines)
            if (len(line) < 80 and line.endswith(':')) or (len(line) < 50 and line.isupper()):
                story.append(Paragraph(line, subheading_style))
                continue
                
            # Check for bullet points or numbered lists
            if line.startswith('•') or line.startswith('-') or re.match(r'^\d+\.', line):
                story.append(Paragraph(line, normal_style))
                story.append(Spacer(1, 0.05*inch))
                continue
                
            # Regular paragraph
            story.append(Paragraph(line, normal_style))
            story.append(Spacer(1, 0.1*inch))
        
        # Add page break between pages
        if page_num != pages_to_process[-1]:  # Not the last page
            story.append(PageBreak())
    
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
        
        # Translate with structure preservation
        translated_text = translate_text_with_llm(extracted_text, preserve_structure=True)
        translated_pages[page_number] = translated_text
    
    # Create PDF with translated text and enhanced structure
    create_enhanced_pdf_with_translated_text(translated_pages, output_pdf_path, page_number)
    
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
        # Translate with structure preservation
        translated_text = translate_text_with_llm(extracted_text, preserve_structure=True)
        translated_pages[page_number] = translated_text
    
    # Create PDF with all translated pages and enhanced structure
    create_enhanced_pdf_with_translated_text(translated_pages, output_pdf_path)
    
    print(f"Translation complete! Saved to: {output_pdf_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Translate Arabic PDF to English PDF file with structure preservation')
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
