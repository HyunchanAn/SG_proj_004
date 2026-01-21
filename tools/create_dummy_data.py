import os
import numpy as np
from PIL import Image

def create_image(filename, color, size=(100, 100)):
    img = Image.new('RGB', size, color=color)
    img.save(filename)
    print(f"Created dummy image: {filename}")

def create_dummy_model(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        f.write(b'DUMMY MODEL CONTENT')
    print(f"Created dummy model: {filename}")

def main():
    # Root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    print(f"Working in: {root_dir}")

    # 1. Create Images
    create_image('test_surface.jpg', (0, 0, 0)) # Black
    create_image('test_water1.jpg', (0, 0, 255)) # Blue
    create_image('test_water2.jpg', (0, 0, 255)) # Blue
    create_image('test_diiodo1.jpg', (128, 128, 128)) # Gray

    # 2. Create Model
    create_dummy_model('models/mobile_sam.pt')

    print(">>> Dummy data generation complete.")

if __name__ == "__main__":
    main()
