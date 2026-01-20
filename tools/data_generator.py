import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_data(num_samples=1000, output_path='training_data.csv'):
    print(f"Generating {num_samples} synthetic samples...")
    
    # Feature Definitions
    materials = ['Metal', 'Plastic', 'Glass', 'Wood']
    finishes = ['Rough', 'Mirror', 'Hairline', 'Matte']
    
    data = []
    
    for _ in range(num_samples):
        # 1. Substrate Features (Random)
        mat = random.choice(materials)
        finish = random.choice(finishes)
        
        # SFE (Surface Free Energy): Metal > Glass > Plastic > Wood broadly, but random variation
        base_sfe = {
            'Metal': 40, 'Glass': 35, 'Plastic': 30, 'Wood': 25
        }[mat]
        sfe = np.random.normal(base_sfe, 5.0) # dyne/cm
        
        # Roughness (Ra): Mirror < Hairline < Matte < Rough
        base_ra = {
            'Mirror': 0.1, 'Hairline': 0.5, 'Matte': 1.5, 'Rough': 3.0
        }[finish]
        roughness = np.max([0, np.random.normal(base_ra, 0.5)])
        
        # 2. Product Specs (Random simulation of products)
        # G' (Storage Modulus): hard vs soft
        g_prime = np.random.randint(5000, 35000) 
        # G'' (Loss Modulus)
        g_double = g_prime * np.random.uniform(0.1, 0.5)
        thickness = np.random.choice([50, 75, 100, 150, 200])
        cross_link = np.random.choice([0, 1, 2]) # 0: Low, 1: Med, 2: High
        
        # 3. Target Label Simulation (The "Physics" Logic)
        # Logic: High SFE + Soft Adhesives (Low G') -> High Holding Time, but Residue Risk
        #        Low SFE + Hard Adhesives (High G') -> Low Holding Time, Clean Removal
        
        # Holding Time Base (Log scale-ish)
        # Match score: Alignment of Surface Energy and Product Wetting (inverse of G')
        wetting_ability = (sfe / 50.0) * (20000 / g_prime) 
        base_time = 24.0 * wetting_ability * (thickness / 100.0)
        
        # Failure Mode Logic
        # If wetting is TOO good (high SFE, low G'), cohesive failure (residue) is likely
        residue_prob = 0.0
        if wetting_ability > 1.2:
            residue_prob = 0.8
        elif wetting_ability > 0.8:
            residue_prob = 0.3
        else:
            residue_prob = 0.05
            
        # Adjust time based on failure mode
        # Cohesive failure usually means it stuck well but failed internally -> Long time provided structure held
        if np.random.random() < residue_prob:
            fail_mode = 0 # Residue (Bad for removal, Good for stick)
            final_time = base_time * 1.5 + np.random.normal(0, 5)
        else:
            fail_mode = 1 # Clean (Adhesion Failure)
            final_time = base_time + np.random.normal(0, 5)
            
        # One-hot encoding for material/finish not strictly needed if we use simple numerical logic, 
        # but for compatibility with real ML features:
        # We will save raw values for now and preprocess in training
        
        data.append({
            'Material': mat,
            'Finish': finish,
            'SFE': sfe,
            'Roughness': roughness,
            'G_prime': g_prime,
            'G_double_prime': g_double,
            'Thickness': thickness,
            'Cross_Link': cross_link,
            'Holding_Time': max(0.1, final_time), # hours
            'Failure_Mode': fail_mode # 0: Residue, 1: Clean
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if not os.path.exists('tools'):
        os.makedirs('tools')
    generate_synthetic_data()
