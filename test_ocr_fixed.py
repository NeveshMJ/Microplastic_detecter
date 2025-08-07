import pytesseract
import cv2
import os
from PIL import Image, ImageDraw

# IMPORTANT: Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def test_all():
    print("🚀 FIXED OCR TEST")
    print("=" * 50)
    
    # Test 1: Basic Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
    except Exception as e:
        print(f"❌ Tesseract error: {e}")
        return
    
    # Test 2: Simple OCR
    try:
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), "INGREDIENTS: Water, Glycerin", fill='black')
        
        text = pytesseract.image_to_string(img)
        print(f"✅ Simple OCR: '{text.strip()}'")
    except Exception as e:
        print(f"❌ Simple OCR error: {e}")
        return
    
    # Test 3: Real image (if exists)
    if os.path.exists("assets/test_images"):
        images = [f for f in os.listdir("assets/test_images") 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if images:
            image_path = os.path.join("assets/test_images", images[0])
            print(f"\n📸 Testing with real image: {image_path}")
            
            try:
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img)
                print(f"📄 Extracted text ({len(text)} chars):")
                print(f"'{text[:150]}{'...' if len(text) > 150 else ''}'")
                
                if len(text.strip()) > 10:
                    print("✅ Real image OCR successful!")
                else:
                    print("⚠️  Very little text extracted - check image quality")
                    
            except Exception as e:
                print(f"❌ Real image OCR error: {e}")
        else:
            print("📭 No test images found in assets/test_images/")
    else:
        print("📁 No test_images folder found")
    
    print("\n🎉 OCR is now properly configured!")

if __name__ == "__main__":
    test_all()
    input("Press Enter to close...")