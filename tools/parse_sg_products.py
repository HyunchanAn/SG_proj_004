import re
import pandas as pd
import numpy as np

def parse_and_update_db(text_file, output_csv):
    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read()

    products = []
    
    # Heuristic: Find explicit product lines or table rows
    # Look for "SG" followed by code
    # Then look for nearby specs.
    
    # Pattern A: "SGS-9715-L Silicone Transparent PET 15 0.085"
    # Regex: (SGS-[\w-]+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\d+)\s+([\d\.]+)
    # Adhesion is group 5, Thickness is group 6
    
    # Pattern B: "SGV-227" on one line, specs later "75 WhitePVC ... 0.10"
    
    lines = content.split('\n')
    current_products = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        # Detect Product Codes
        # Matches SGV-227, SGO-403TT, SGS-9715-L
        codes = re.findall(r'(SG[A-Z]-[\w\d/-]+)', line)
        if codes:
            # If line is mostly just codes, assume specific header
            # But sometimes codes are in a table row: "SGS-9701-L Silicone..."
            
            # Check if this line ALSO has data
            # Data usually has numbers like 0.05, 1000, etc.
            if re.search(r'\d+\s+0\.\d+', line):
                # Inline data
                # Try to parse row
                # e.g. SGS-9701-L Silicone Transparent PET 1 0.085
                parts = line.split()
                # Find the float for thickness (0.xxx)
                thick = 0.05
                adhesion = 100
                
                # Simple parser: look for number < 1 (thickness) and number > 1 (adhesion)
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                nums = [float(n) for n in nums]
                
                # heuristics
                # Thickness is usually 0.03 ~ 0.20
                thickness_cands = [n for n in nums if 0.02 < n < 0.5]
                # Adhesion is usually > 1 (gf)
                adhesion_cands = [n for n in nums if n > 1 and n < 3000]
                
                if thickness_cands: thick = thickness_cands[0]
                if adhesion_cands: adhesion = adhesion_cands[0]
                
                for code in codes:
                    # Clean code
                    for c in code.split('/'): # SGV-247/257
                        products.append({
                            'ID': c.strip(),
                            'Adhesion': adhesion,
                            'Thickness': thick * 1000 # Convert mm to micron roughly? No wait, DB uses micron? 
                            # Previous Dummy DB used Thickness=50 (microns). Text has 0.05 mm = 50 microns.
                            # So multiply by 1000.
                        })
            else:
                # Header codes, look ahead for data
                # Store these codes
                current_products = []
                for code in codes:
                    for c in code.split('/'):
                        current_products.append(c.strip())
        
        # If we have current_products waiting for data
        elif current_products and re.search(r'^\d+', line):
            # Line starts with number (Adhesion)
            # "75 WhitePVC ..."
            try:
                parts = line.split()
                adhesion = float(parts[0])
                
                # Look for thickness like 0.10
                thick = 50.0 # Default
                nums = re.findall(r"0\.\d+", line)
                if nums:
                    thick = float(nums[0]) * 1000 # mm to micron
                
                for p in current_products:
                    products.append({
                        'ID': p,
                        'Adhesion': adhesion,
                        'Thickness': thick
                    })
                current_products = [] # Reset
            except:
                pass

    # Remove duplicates
    unique_products = {}
    for p in products:
        unique_products[p['ID']] = p
    
    products = list(unique_products.values())
    print(f"Parsed {len(products)} products.")
    
    # Augment with Rheology (Simulated from Adhesion)
    # High Adhesion -> Soft (Low G')
    # Low Adhesion -> Hard (High G')
    final_data = []
    
    for p in products:
        adh = p['Adhesion']
        # Mapping: 10gf -> 50000, 2000gf -> 5000
        # Log scale mapping might be better
        # G' = 500,000 / (adh + 10) Is roughly 
        # If adh=10, G'=25k? No.
        # Let's use linear interpolation for simplicity in prototype
        # Max Adh ~ 1500 -> G' 5000
        # Min Adh ~ 10 -> G' 35000
        
        # Slope = (35000 - 5000) / (10 - 1500) = 30000 / -1490 ~ -20
        g_prime = 35000 + (adh - 10) * (-20)
        g_prime = max(3000, g_prime) # Clamp
        
        g_double = g_prime * 0.4 # Viscous
        
        final_data.append({
            'ID': p['ID'],
            'Name': p['ID'], # Use ID as name
            'G_prime': int(g_prime),
            'G_double_prime': int(g_double),
            'Thickness': int(p['Thickness']),
            'Cross_Link_Density': 'Med', # Default
            'Target_Spec': adh # Keep original adhesion as spec
        })
        
    df = pd.DataFrame(final_data)
    df.to_csv(output_csv, index=False)
    print(f"Database saved to {output_csv}")
    print(df.head())

if __name__ == "__main__":
    parse_and_update_db("SG_products/extracted_text.txt", "assets/sg_product_db.csv")
