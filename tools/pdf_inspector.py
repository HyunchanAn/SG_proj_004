import os
from pypdf import PdfReader

def extract_pdf_data(directory):
    products = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            path = os.path.join(directory, filename)
            print(f"Processing: {filename}")
            try:
                reader = PdfReader(path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                # Simple heuristic to print some content to see if we can parse it
                print(f"--- Content of {filename} (First 500 chars) ---")
                print(text[:500])
                print("------------------------------------------------")
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    directory = "SG_products"
    full_text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            path = os.path.join(directory, filename)
            try:
                reader = PdfReader(path)
                full_text += f"\n=== FILE: {filename} ===\n"
                for page in reader.pages:
                    full_text += page.extract_text() + "\n"
            except Exception as e:
                full_text += f"Error reading {filename}: {e}\n"
    
    with open(os.path.join(directory, "extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write(full_text)
    print("Done. Saved to SG_products/extracted_text.txt")
