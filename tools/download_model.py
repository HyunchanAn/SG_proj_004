import requests
import os
import sys

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

def verify_model(path):
    print(f"Verifying model at {path}...")
    try:
        import torch
        # Just try loading it with torch.load to see if it's a valid pt file
        # We map to cpu to avoid cuda errors if not setup
        checkpoint = torch.load(path, map_location='cpu')
        print(f"Model loaded successfully! Keys: {list(checkpoint.keys())[:5]}")
        return True
    except Exception as e:
        print(f"Model verification failed: {e}")
        return False

if __name__ == "__main__":
    url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    target_path = os.path.join("models", "mobile_sam.pt")
    
    # Ensure models dir exists
    os.makedirs("models", exist_ok=True)
    
    if download_file(url, target_path):
        if verify_model(target_path):
            print(">>> MODEL SETUP COMPLETE: Ready for Real Mode.")
        else:
            print(">>> MODEL DOWNLOADED BUT INVALID.")
            sys.exit(1)
    else:
        sys.exit(1)
