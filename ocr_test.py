"""
Library to install
pip install C:\code\ocr\whl_files\paddleocr-2.9.1-py3-none-any.whl
pip install C:\code\ocr\whl_files\paddlepaddle-2.6.2-cp312-cp312-win_amd64.whl
pip install PyMuPDF==1.24.11
"""

# ## Simple ocr
# from paddleocr import PaddleOCR
# from pathlib import Path
# # image_folder = Path("C:/code/ocr/images")

# no_text_image = "C:/code/ocr/images/no_text.png"
# button_image = "C:/code/ocr/images/button_image.png"
# table_image = "C:/code/ocr/images/table_image.png"
# ocr = PaddleOCR(use_angle_cls=True, lang="en",)
# result = ocr.ocr(str(table_image), cls=True)
# extracted_text = ""
# for line in result:
#     if not line:
#         continue
#     for word_info in line:
#         extracted_text += word_info[1][0] + " " 

# print(extracted_text.strip() + "\n\n")




import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import os
from PIL import Image
import io
from datetime import datetime

class PDFProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.processing_date = "2025-03-03 09:02:23"
        self.user_login = "Navanit-git"
        
    def process_page(self, page, page_num, output_dir):
        """Process a single page and return its content in order"""
        blocks = page.get_text("blocks")
        images = page.get_images()
        
        # List to store all content with their y-coordinates
        all_content = []
        
        # First, process text blocks
        for b in blocks:
            y_coord = b[1]  # y-coordinate of text block
            text = b[4].strip()
            if text:
                all_content.append((y_coord, text))
                print(f"Added text block at y-coord {y_coord}: {text[:50]}...")
        
        # Process images
        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Get image location
                image_info = page.get_image_info(xref)
                y_coord = image_info[0]['bbox'][1] if image_info else 0
                
                # Process image with OCR
                image = Image.open(io.BytesIO(image_bytes))
                temp_image_path = os.path.join(output_dir, f'temp_image_p{page_num}_{img_index + 1}.png')
                image.save(temp_image_path)
                
                result = self.ocr.ocr(temp_image_path, cls=True)
                
                image_text = ""
                if result:
                    for line in result:
                        if line:
                            for item in line:
                                if item and len(item) >= 2 and item[1] and item[1][0]:
                                    image_text += f"{item[1][0]} "
                
                # If we found text in the image, add it to all_content
                if image_text.strip():
                    print(f"Added image text at y-coord {y_coord}: {image_text.strip()[:50]}...")
                    all_content.append((y_coord, f"[Image Text: {image_text.strip()}]"))
                
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                    
            except Exception as e:
                print(f"Error processing image {img_index + 1} on page {page_num}: {str(e)}")
                continue
        
        # Sort all content by y-coordinate
        all_content.sort(key=lambda x: x[0])
        
        # Return only the text content in order
        return [text for _, text in all_content]
        
    def process_pdf(self, pdf_path, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        pdf_document = fitz.open(pdf_path)
        
        all_pages_content = []
        
        # Process each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            print(f"\nProcessing page {page_num + 1}")
            
            # Process the page
            page_content = self.process_page(page, page_num + 1, output_dir)
            
            if page_content:
                # Add page marker
                all_pages_content.append(f"\nPage {page_num + 1}:")
                # Add all content for this page
                all_pages_content.extend(page_content)
        
        # Write to file
        output_path = os.path.join(output_dir, 'extracted_text.txt')
        with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
            for line in all_pages_content:
                f.write(f"{line}\n")
        
        pdf_document.close()
        print(f"\nProcessing complete. Output saved to: {output_path}")
        return output_path

def main():
    try:
        pdf_processor = PDFProcessor()
        pdf_path = "C:/code/ocr/online_banking_guide_small.pdf"  # Replace with your PDF path
        output = pdf_processor.process_pdf(pdf_path)
        print(f"Successfully processed PDF. Output saved to: {output}")
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main()
