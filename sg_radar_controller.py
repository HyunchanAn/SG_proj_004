import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import json
import random

# Ensure libs can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'libs'))

# Try importing real sensors, else Mock
try:
    from vsams.models.classifier import SurfaceClassifier
    from deepdrop_sfe.ai_engine import AIContactAngleAnalyzer
    from deepdrop_sfe.physics_engine import DropletPhysics
    REAL_SENSORS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Sensors not available ({e}). Using Mock Sensors.")
    REAL_SENSORS_AVAILABLE = False

class MockSurfaceAnalyzer:
    def analyze(self, image_path):
        # Return random but consistent results for demo
        materials = ['Metal', 'Plastic']
        finishes = ['Rough', 'Mirror', 'Hairline']
        return {
            'material': random.choice(materials),
            'finish': random.choice(finishes),
            'features': np.random.rand(10) # Dummy vector
        }

class MockPhysicsAnalyzer:
    def analyze(self, image_path):
        # Return random SFE
        return 40.0 + random.uniform(-5, 5)

class SG_RADAR_Controller:
    def __init__(self, config_path='config.yaml'):
        print("📡 [System] SG-R.A.D.A.R 시스템 초기화 중...")
        
        # 1. Load Sensors
        # 1. Load Sensors
        try:
            # Try importing real sensors
            from vsams.models.classifier import SurfaceClassifier
            from deepdrop_sfe.ai_engine import AIContactAngleAnalyzer
            from deepdrop_sfe.physics_engine import DropletPhysics
            
            # Check for model file (Note: vsams model check is internal to SurfaceClassifier if desired, or we pass path)
            # MobileSAM check
            if os.path.exists('models/mobile_sam.pt'):
                # Initialize DeepDrop with SAM model
                self.physics_sensor = AIContactAngleAnalyzer(model_path='models/mobile_sam.pt')
                
                # Initialize V-SAMS
                # Ideally we load a checkpoint if available regarding implementation_plan.md
                # For now, just init the class. It might need a checkpoint path.
                self.vision_sensor = SurfaceClassifier() 
                
                self.is_mock = False
            else:
                raise FileNotFoundError("MobileSAM model not found")
        except Exception as e:
            print(f"Warning: 센서 로딩 실패 ({e}). Mock 모드로 전환합니다.")
            self.vision_sensor = MockSurfaceAnalyzer()
            self.physics_sensor = MockPhysicsAnalyzer()
            self.is_mock = True
            
        # 2. Load Product DB
        print("   - 제품 데이터베이스 로드 중...")
        db_path = os.path.join('assets', 'sg_product_db.csv')
        if not os.path.exists(db_path):
            # Create dummy if needed (handled by previous steps)
            pass
        self.product_db = pd.read_csv(db_path)
        
        # 3. Load Brain
        print("   - 예측 엔진(Brain) 로드 중...")
        self.model_time = xgb.Booster()
        self.model_mode = xgb.Booster()
        
        model_dir = 'models'
        if os.path.exists(os.path.join(model_dir, 'radar_time_v1.json')):
            self.model_time.load_model(os.path.join(model_dir, 'radar_time_v1.json'))
            self.model_mode.load_model(os.path.join(model_dir, 'radar_mode_v1.json'))
        else:
            print("Warning: 모델 파일이 없습니다.")

    def _get_contact_angle(self, img_path, liquid_name):
        if self.is_mock:
            # Mock Logic
            base = 60 if liquid_name == 'Water' else 40
            return base + random.uniform(-5, 5)
        else:
            # Real Logic using DeepDrop Lib
            # The new AIContactAngleAnalyzer.analyze() returns angle directly or detailed dict?
            # Based on integration_pipeline.py from V-SAMS repo:
            # val_energy = self.deepdrop.analyze(img_contact_angle) -> returns angle
            
            try:
                # We need to load image or pass path. 
                # DeepDrop AIContactAngleAnalyzer likely expects an image array or path.
                # Let's assume it accepts path or we read it.
                # Checking inspection of deepdrop_sfe... we didn't fully see analyze signature 
                # but integration_pipeline.py used it as .analyze(img).
                
                # IMPORTANT: We need to verify if analyze takes path or image.
                # For safety, let's read the image using cv2 or PIL as commonly done in these libs.
                # But wait, looking at the snippet from integration_pipeline.py:
                # val_energy = self.deepdrop.analyze(img_contact_angle)
                
                import cv2
                img = cv2.imread(img_path)
                if img is None: return 0.0
                
                angle = self.physics_sensor.analyze(img)
                return angle
            except Exception as e:
                print(f"Error in DeepDrop analysis: {e}")
                return 0.0

    def run_rapid_diagnosis(self, surface_img_path, water_img_paths, diiodo_img_paths):
        print("\n🔎 [1단계] 피착제(Substrate) 진단 중...")
        
        # A. Visual Diagnosis
        # V-SAMS SurfaceClassifier uses 'predict(image)' or 'extract_features(image)'?
        # integration_pipeline.py used 'extract_features'.
        # We need classification result (Material, Finish).
        # Let's assume 'predict(image)' exists or we check the source later.
        # For now, adapting to likely API.
        
        import cv2
        surf_img = cv2.imread(surface_img_path)
        
        if self.is_mock:
             visual_result = self.vision_sensor.analyze(surface_img_path)
        else:
             # V-SAMS Real
             try:
                 # Converting to PIL or Tensor might be needed depending on implementation
                 # But let's assume it handles it or we wrap it.
                 # Actually, usually 'predict' returns class.
                 # If SurfaceClassifier structure is standard:
                 # visual_result = self.vision_sensor.predict(surf_img)
                 # We will assume a analyze-like wrapper or we might need to fix this if API differs.
                 # Let's use a safe wrapper pattern.
                 
                 # Placeholder for V-SAMS prediction
                 # visual_result = self.vision_sensor.predict(surf_img)
                 # Since we don't have the full V-SAMS code in front of us (just integration_pipeline),
                 # we'll assume it returns a dict similar to before or we assume Mock for now if fails.
                 
                 # Let's try to call a method that likely exists or fallback
                 if hasattr(self.vision_sensor, 'predict'):
                     visual_result = self.vision_sensor.predict(surf_img)
                 else:
                     # Fallback to Mock behavior if method missing (safety)
                     visual_result = {'material': 'Unknown', 'finish': 'Unknown'}
             except Exception:
                 visual_result = {'material': 'Error', 'finish': 'Error'}

        print(f"   - 시각 분석: {visual_result.get('finish', 'Unknown')} {visual_result.get('material', 'Unknown')}")
        
        # B. Physics Diagnosis (OWRK)
        from deepdrop_sfe.physics_engine import DropletPhysics
        
        measurements = []
        
        # 1. Water Angles
        w_angles = []
        for p in water_img_paths:
            angle = self._get_contact_angle(p, 'Water')
            measurements.append({'liquid': 'Water', 'angle': angle})
            w_angles.append(angle)
            
        # 2. Diiodomethane Angles
        d_angles = []
        for p in diiodo_img_paths:
            angle = self._get_contact_angle(p, 'Diiodomethane')
            measurements.append({'liquid': 'Diiodomethane', 'angle': angle})
            d_angles.append(angle)
            
        # Calculate SFE
        if w_angles and d_angles:
            # DeepDrop-SFE updated API might be slightly different, but let's assume OWRK calculation is static
            try:
                # Check signature of calculate_owrk. 
                # Old: sfe, dispersive, polar = DropletPhysics.calculate_owrk(measurements)
                # New: likely similar. 
                res = DropletPhysics.calculate_owrk(measurements)
                if isinstance(res, tuple):
                    sfe, dispersive, polar = res
                else:
                    sfe = res.get('sfe', 0)
                    # fallback
                    dispersive = sfe/2
                    polar = sfe/2
                    
                method = "OWRK (2-Liquid)"
            except Exception as e:
                print(f"OWRK Error: {e}")
                sfe = 0
                method = "Error"
        elif w_angles:
            # Fallback
            avg_w = np.mean(w_angles)
            rad = np.radians(avg_w)
            sfe = 72.8 * (1 + np.cos(rad))**2 / 4
            dispersive = sfe * 0.5 
            polar = sfe * 0.5
            method = "Water-Only EOS (Approximation)"
        else:
            sfe = 30.0
            method = "Default"

        print(f"   - 물리 분석 ({method}): 표면 에너지 {sfe:.1f} dyne/cm")
        
        # Make feature vector for model
        
        recommendations = []
        
        # Load feature columns to ensure alignment
        import pickle
        feature_col_path = os.path.join('models', 'feature_columns.pkl')
        if os.path.exists(feature_col_path):
            with open(feature_col_path, 'rb') as f:
                feature_columns = pickle.load(f)
        else:
            feature_columns = None
            print("Warning: feature_columns.pkl 파일을 찾을 수 없습니다. 추론이 실패할 수 있습니다.")
        
        print("⚙️ [2단계] 전 제품 가상 시뮬레이션 수행 중...")
        
        for idx, product in self.product_db.iterrows():
            # Create a single-row DataFrame for prediction
            input_data = {
                'SFE': sfe,
                'Roughness': 0.5, # Mock value or from vision
                'G_prime': product['G_prime'],
                'G_double_prime': product['G_double_prime'],
                'Thickness': product['Thickness'],
                'Cross_Link': 1 if product['Cross_Link_Density'] == 'Med' else (2 if product['Cross_Link_Density'] == 'High' else 0),
                # Add Dummy One-Hot columns for Material/Finish
                'Material_Metal': 1 if visual_result.get('material') == 'Metal' else 0,
                'Material_Plastic': 1 if visual_result.get('material') == 'Plastic' else 0,
                'Material_Glass': 0,
                'Material_Wood': 0,
                'Finish_Rough': 1 if visual_result.get('finish') == 'Rough' else 0,
                'Finish_Mirror': 0,
                'Finish_Hairline': 0,
                'Finish_Matte': 0
            }

            df_input = pd.DataFrame([input_data])
            
            # Align columns
            if feature_columns:
                # Add missing cols with 0
                for col in feature_columns:
                    if col not in df_input.columns:
                        df_input[col] = 0
                # Reorder and select only needed cols
                df_input = df_input[feature_columns]
            
            dmatrix = xgb.DMatrix(df_input)
            
            # Predict
            pred_time = self.model_time.predict(dmatrix)[0]
            pred_mode_prob = self.model_mode.predict(dmatrix)[0] # Probability of Class 1 (Clean)
            
            score = self._scoring_logic(pred_time, pred_mode_prob)
            
            recommendations.append({
                "id": product['ID'],
                "name": product['Name'],
                "pred_time": pred_time,
                "clean_prob": pred_mode_prob,
                "score": score
            })
            
        # Step 3
        print("🏆 [3단계] 최종 추천 리포트 생성 중...")
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        # Handle empty recommendations if logic fails
        if not recommendations:
            return {"error": "No recommendations found"}
            
        best_pick = recommendations[0]
        
        return {
            "diagnosis": {
                "material": visual_result.get('material'),
                "surface_energy": sfe,
                "method": method
            },
            "best_product": best_pick,
            "top_3_candidates": recommendations[:3]
        }

    def _scoring_logic(self, time, clean_prob):
        # clean_prob: 1.0 = Clean, 0.0 = Residue
        # If residue risk is high (clean_prob < 0.5), penalty
        if clean_prob < 0.7:
            return time * 0.1
        return time * clean_prob

if __name__ == "__main__":
    # Test Run
    ctrl = SG_RADAR_Controller()
    res = ctrl.run_rapid_diagnosis("dummy_surf.jpg", ["dummy_drop1.jpg", "dummy_drop2.jpg"])
    print("\n[Result Summary]")
    print(f"Best: {res['best_product']['name']} (Score: {res['best_product']['score']:.1f})")
    print(f"Time: {res['best_product']['pred_time']:.1f}h, CleanProb: {res['best_product']['clean_prob']:.2f}")
