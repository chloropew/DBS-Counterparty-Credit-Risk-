import sys
import os

# Remove current directory from path to avoid circular imports
if '' in sys.path:
    sys.path.remove('')
if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())

def extract_dbs_data(pdf_path, output_excel='dbs_exposure_data.xlsx'):
    """
    Extract exposure data from DBS Annual Report with enhanced table detection
    """
    
    import pdfplumber
    import pandas as pd
    
    print("="*70)
    print("DBS EXPOSURE DATA EXTRACTOR - ENHANCED VERSION")
    print("="*70)
    print(f"PDF: {pdf_path}")
    print(f"Output: {output_excel}\n")
    
    # Keywords to search for
    keywords = {
        'geographic': ['geographic', 'geographical distribution', 'by country', 'regional'],
        'industry': ['industry', 'sector exposure', 'business segment'],
        'credit_quality': ['credit quality', 'non-performing', 'impaired', 'npl'],
        'asset_class': ['asset class', 'loan type', 'exposure by type']
    }
    
    found_pages = {}
    all_tables = {}
    
    # Step 1: Scan for relevant pages
    print("Step 1: Scanning PDF for exposure tables...")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}\n")
            
            for i, page in enumerate(pdf.pages, 1):
                if i % 20 == 0:
                    print(f"  Scanned {i}/{total_pages} pages...")
                
                text = page.extract_text()
                if text:
                    text_lower = text.lower()
                    
                    for category, search_terms in keywords.items():
                        if any(term in text_lower for term in search_terms):
                            if category not in found_pages:
                                found_pages[category] = []
                            if i not in found_pages[category]:
                                found_pages[category].append(i)
    
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return
    
    if not found_pages:
        print("❌ No exposure data found in PDF")
        return
    
    print("\n✓ Found exposure data on these pages:")
    for category, pages in found_pages.items():
        print(f"  {category}: pages {pages[:5]}")
    
    # Step 2: Extract tables with multiple strategies
    print("\nStep 2: Extracting tables with enhanced detection...")
    
    # Table detection settings to try
    table_settings = [
        {},  # Default settings
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
        },
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
        },
        {
            "vertical_strategy": "explicit",
            "horizontal_strategy": "explicit",
            "explicit_vertical_lines": [],
            "explicit_horizontal_lines": [],
        }
    ]
    
    with pdfplumber.open(pdf_path) as pdf:
        for category, pages in found_pages.items():
            for page_num in pages[:3]:
                try:
                    page = pdf.pages[page_num - 1]
                    
                    # Try different extraction strategies
                    tables = None
                    for settings in table_settings:
                        try:
                            tables = page.extract_tables(table_settings=settings)
                            if tables:
                                break
                        except:
                            continue
                    
                    if tables:
                        for idx, table in enumerate(tables):
                            if not table or len(table) < 2:
                                continue
                            
                            try:
                                # Try to create dataframe
                                if table[0] and any(cell for cell in table[0] if cell):
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                else:
                                    df = pd.DataFrame(table)
                                
                                # Clean
                                df = df.dropna(how='all').dropna(axis=1, how='all')
                                df = df.loc[:, df.columns.notna()]
                                
                                if len(df) > 1:
                                    table_name = f"{category}_p{page_num}_t{idx+1}"
                                    all_tables[table_name] = df
                                    print(f"  ✓ {table_name}: {df.shape[0]} rows × {df.shape[1]} cols")
                            except Exception as e:
                                continue
                    else:
                        # If no tables found, try extracting as text and look for tabular data
                        text = page.extract_text()
                        if text:
                            lines = text.split('\n')
                            # Look for lines with multiple columns (spaces or tabs)
                            tabular_lines = [line for line in lines if line.count('  ') >= 2 or '\t' in line]
                            
                            if len(tabular_lines) > 5:
                                print(f"  ⚠ Page {page_num}: Found text data but no structured tables")
                                print(f"    Try manual extraction or check if page contains images")
                
                except Exception as e:
                    print(f"  ⚠ Error on page {page_num}: {str(e)[:80]}")
    
    if not all_tables:
        print("\n" + "="*70)
        print("❌ NO TABLES EXTRACTED")
        print("="*70)
        print("\nPossible reasons:")
        print("1. Tables are embedded as images (need OCR)")
        print("2. Tables use complex formatting")
        print("3. PDF is scanned (not text-based)")
        print("\n💡 RECOMMENDED NEXT STEPS:")
        print("  - Try the manual extraction method below")
        print("  - Or share the specific page numbers and I'll help extract manually")
        return
    
    # Step 3: Save to Excel
    print(f"\nStep 3: Saving to {output_excel}...")
    
    try:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            for sheet_name, df in all_tables.items():
                safe_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS!")
        print(f"{'='*70}")
        print(f"Extracted {len(all_tables)} tables")
        print(f"Saved to: {output_excel}")
        print(f"{'='*70}\n")
        
        # Preview
        print("Preview of extracted tables:")
        for name, df in list(all_tables.items())[:2]:
            print(f"\n{name}:")
            print(df.head(3).to_string())
            print("-" * 70)
        
        if len(all_tables) > 2:
            print(f"\n... and {len(all_tables) - 2} more tables in the Excel file")
    
    except Exception as e:
        print(f"❌ Error saving Excel: {e}")


def extract_text_from_page(pdf_path, page_number, output_txt='page_text.txt'):
    """Extract all text from a page to inspect manually"""
    import pdfplumber
    
    print(f"Extracting text from page {page_number}...")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number > len(pdf.pages):
                print(f"❌ Page {page_number} doesn't exist")
                return
            
            page = pdf.pages[page_number - 1]
            text = page.extract_text()
            
            if text:
                with open(output_txt, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"✅ Text saved to {output_txt}")
                print(f"\nPreview (first 1000 characters):")
                print("-" * 70)
                print(text[:1000])
                print("-" * 70)
            else:
                print(f"❌ No text found (page might be scanned image)")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def diagnose_page(pdf_path, page_number):
    """Diagnose what's on a specific page"""
    import pdfplumber
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSING PAGE {page_number}")
    print(f"{'='*70}\n")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            
            # Check for text
            text = page.extract_text()
            print(f"Text length: {len(text) if text else 0} characters")
            
            # Check for tables
            tables = page.extract_tables()
            print(f"Tables found: {len(tables) if tables else 0}")
            
            # Check for images
            images = page.images
            print(f"Images found: {len(images)}")
            
            # Check page dimensions
            print(f"Page size: {page.width} x {page.height}")
            
            if text:
                print(f"\nFirst 500 characters of text:")
                print("-" * 70)
                print(text[:500])
                print("-" * 70)
            
            if tables:
                print(f"\nFirst table preview:")
                print(tables[0][:5])
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    PDF_PATH = "DBS Annual Report 2024.pdf"
    OUTPUT_FILE = "dbs_exposure_data.xlsx"
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF file not found: {PDF_PATH}")
        print(f"\nPDF files in current directory:")
        for f in os.listdir():
            if f.lower().endswith('.pdf'):
                print(f"  - {f}")
    else:
        # Try automatic extraction
        extract_dbs_data(PDF_PATH, OUTPUT_FILE)
        
        # If extraction failed, diagnose specific pages
        print("\n" + "="*70)
        print("DIAGNOSTIC MODE")
        print("="*70)
        print("\nDiagnosing key pages found earlier...")
        
        for page_num in [7, 68, 86]:  # Pages that were found
            diagnose_page(PDF_PATH, page_num)
            print("\n")