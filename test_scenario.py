from sg_radar_controller import SG_RADAR_Controller
import os

def run_test():
    print(">>> Starting System Verification Test...")
    
    # 1. Initialize Controller
    try:
        radar = SG_RADAR_Controller()
    except Exception as e:
        print(f"FAILED to initialize controller: {e}")
        return

    # 2. Test Image Paths (User Provided)
    image_dir = "test_images"
    surface_img = os.path.join(image_dir, "surface_test_image (1).jpg")
    water_imgs = [
        os.path.join(image_dir, "drop_test_image (1).jpg"),
        os.path.join(image_dir, "drop_test_image (2).jpg")
    ]
    diiodo_imgs = [
        os.path.join(image_dir, "drop_test_image (3).jpg")
    ] 
    
    # 3. Run Diagnosis
    try:
        # Now expects 3 arguments
        result = radar.run_rapid_diagnosis(surface_img, water_imgs, diiodo_imgs)
        
        print("\n>>> 테스트 결과:")
        print(f"   진단 정보: {result['diagnosis']}")
        print(f"   Best Product: {result['best_product']['name']}")
        print(f"   Score: {result['best_product']['score']:.4f}")
        print(">>> VERIFICATION SUCCESS")
        
    except Exception as e:
        print(f"FAILED during diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
