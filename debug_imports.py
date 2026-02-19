import sys
import os
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), 'libs'))

print("1. Testing V-SAMS import...")
try:
    from vsams.models.classifier import SurfaceClassifier
    print("   SUCCESS")
except Exception:
    traceback.print_exc()

print("\n2. Testing DeepDrop-SFE import...")
try:
    from deepdrop_sfe.ai_engine import AIContactAngleAnalyzer
    print("   SUCCESS")
except Exception:
    traceback.print_exc()

print("\n3. Testing Model Load...")
try:
    import torch
    model_path = 'models/mobile_sam.pt'
    if os.path.exists(model_path):
        print(f"   Model file exists: {os.path.getsize(model_path)} bytes")
        # Try loading
        from ultralytics.nn.tasks import attempt_load_weights
        # Note: The code in controller uses AIContactAngleAnalyzer which internally uses something. 
        # Let's verify what deepdrop uses.
    else:
        print("   Model file missing!")
except Exception:
    traceback.print_exc()
