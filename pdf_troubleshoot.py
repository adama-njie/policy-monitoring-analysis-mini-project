"""
PDF Troubleshooting & OCR Helper
=================================
This script helps diagnose and fix problematic PDF files.
"""

import PyPDF2
from pathlib import Path

def diagnose_pdf(pdf_path):
    """
    Diagnose issues with a PDF file.
    """
    print(f"\n{'='*60}")
    print(f"Diagnosing: {pdf_path.name}")
    print(f"{'='*60}")
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check encryption
            if pdf_reader.is_encrypted:
                print("❌ Status: ENCRYPTED")
                print("   Solution: PDF needs to be decrypted or password provided")
                print("   Install: pip install pycryptodome")
                return False
            
            # Get page count
            num_pages = len(pdf_reader.pages)
            print(f"✓ Pages: {num_pages}")
            
            # Try to extract text from first few pages
            total_text = ""
            for i in range(min(3, num_pages)):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    total_text += page_text
                except Exception as e:
                    print(f"⚠️  Error extracting page {i+1}: {str(e)}")
            
            # Analyze extraction quality
            if len(total_text.strip()) < 50:
                print(f"❌ Status: POOR TEXT EXTRACTION ({len(total_text)} chars)")
                print("   Likely cause: Scanned/image-based PDF")
                print("   Solution: Requires OCR (Optical Character Recognition)")
                print("\n   OCR Options:")
                print("   1. Online: Use Adobe Acrobat DC or smallpdf.com")
                print("   2. Python: Install pytesseract + pdf2image")
                print("      pip install pytesseract pdf2image")
                print("      Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
                return False
            else:
                print(f"✓ Status: GOOD ({len(total_text)} chars extracted)")
                print(f"   Preview: {total_text[:150]}...")
                return True
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def check_all_pdfs(directory):
    """
    Check all PDFs in a directory.
    """
    pdf_dir = Path(directory)
    pdfs = list(pdf_dir.glob("*.pdf"))
    
    if not pdfs:
        print(f"No PDFs found in {directory}")
        return
    
    print(f"\nFound {len(pdfs)} PDF files. Checking each...")
    
    good_pdfs = []
    bad_pdfs = []
    
    for pdf_path in pdfs:
        is_good = diagnose_pdf(pdf_path)
        if is_good:
            good_pdfs.append(pdf_path.name)
        else:
            bad_pdfs.append(pdf_path.name)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Good PDFs: {len(good_pdfs)}")
    print(f"❌ Problem PDFs: {len(bad_pdfs)}")
    
    if bad_pdfs:
        print("\nProblematic files:")
        for pdf in bad_pdfs:
            print(f"  - {pdf}")
        print("\nRecommendations:")
        print("1. For encrypted PDFs:")
        print("   - Install pycryptodome: pip install pycryptodome")
        print("   - Or decrypt PDFs manually first")
        print("\n2. For scanned PDFs:")
        print("   - Use OCR tool (Adobe, smallpdf, etc.)")
        print("   - Or install pytesseract for Python OCR")
        print("   - Alternatively, skip these documents for now")

# Quick OCR function (requires pytesseract and pdf2image)
def ocr_pdf_simple(pdf_path, output_path):
    """
    Simple OCR for scanned PDFs.
    Requires: pip install pytesseract pdf2image pillow
    And Tesseract installed on system.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        print(f"Converting {pdf_path.name} to images...")
        images = convert_from_path(pdf_path)
        
        print(f"Performing OCR on {len(images)} pages...")
        full_text = ""
        
        for i, image in enumerate(images):
            print(f"  Processing page {i+1}/{len(images)}...")
            text = pytesseract.image_to_string(image)
            full_text += f"\n\n--- Page {i+1} ---\n\n{text}"
        
        # Save to text file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"✓ OCR complete! Saved to: {output_path}")
        return full_text
        
    except ImportError:
        print("❌ OCR libraries not installed.")
        print("Install with: pip install pytesseract pdf2image pillow")
        print("Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        return None
    except Exception as e:
        print(f"❌ OCR failed: {str(e)}")
        return None

if __name__ == "__main__":
    import sys
    
    # Check all PDFs in raw_documents
    check_all_pdfs("raw_documents")
    
    print("\n" + "="*60)
    print("To process a specific problematic PDF with OCR:")
    print("  python pdf_troubleshoot.py ocr <filename.pdf>")
    print("="*60)