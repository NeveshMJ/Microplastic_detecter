# test_ocr_step_by_step.py
"""
Complete OCR testing script to verify everything works properly
"""

import os
import cv2
import pytesseract
from PIL import Image

def test_tesseract_installation():
    """Test 1: Check if Tesseract is installed and accessible"""
    print("🔧 TEST 1: Checking Tesseract Installation")
    print("-" * 50)
    
    # Set the tesseract path for Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    try:
        # Test basic tesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        print("✅ Tesseract is properly installed!")
        return True
    except Exception as e:
        print(f"❌ Tesseract installation error: {e}")
        print("\n💡 Solutions:")
        print("   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("   - Add this line to your code:")
        print("     pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
        return False

def test_simple_text_recognition():
    """Test 2: Test OCR with a simple text image"""
    print("\n🔤 TEST 2: Simple Text Recognition")
    print("-" * 50)
    
    # Set the tesseract path for Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    try:
        # Create a simple test image with text
        from PIL import Image, ImageDraw, ImageFont
        
        # Create white image
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add simple text
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 30), "INGREDIENTS: Water, Glycerin", fill='black', font=font)
        
        # Save test image
        img.save("test_simple.png")
        
        # Test OCR on it
        text = pytesseract.image_to_string(img)
        print(f"📝 Created test image with: 'INGREDIENTS: Water, Glycerin'")
        print(f"🔍 OCR extracted: '{text.strip()}'")
        
        if "ingredients" in text.lower() or "water" in text.lower():
            print("✅ Simple OCR test passed!")
            os.remove("test_simple.png")  # Clean up
            return True
        else:
            print("⚠️  OCR extracted text but accuracy needs improvement")
            return False
            
    except Exception as e:
        print(f"❌ Simple OCR test failed: {e}")
        return False

def test_image_loading(image_path):
    """Test 3: Check if image can be loaded"""
    print(f"\n📸 TEST 3: Image Loading Test")
    print("-" * 50)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("💡 Please add a test image to assets/test_images/")
        return False
    
    try:
        # Test with OpenCV
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print("❌ OpenCV couldn't load image")
            return False
        
        print(f"✅ OpenCV loaded image: {img_cv.shape}")
        
        # Test with PIL
        img_pil = Image.open(image_path)
        print(f"✅ PIL loaded image: {img_pil.size}, mode: {img_pil.mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image loading error: {e}")
        return False

def test_raw_ocr(image_path):
    """Test 4: Raw OCR without preprocessing"""
    print(f"\n🔍 TEST 4: Raw OCR Test")
    print("-" * 50)
    
    # Set the tesseract path for Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    try:
        # Load image
        img = Image.open(image_path)
        
        # Basic OCR
        text = pytesseract.image_to_string(img)
        
        print(f"📄 Raw extracted text ({len(text)} characters):")
        print(f"'{text[:200]}{'...' if len(text) > 200 else ''}'")
        
        if len(text.strip()) > 10:
            print("✅ OCR extracted substantial text")
            return True, text
        else:
            print("⚠️  OCR extracted very little text - image might need preprocessing")
            return False, text
            
    except Exception as e:
        print(f"❌ Raw OCR failed: {e}")
        return False, ""

def test_preprocessed_ocr(image_path):
    """Test 5: OCR with image preprocessing"""
    print(f"\n⚡ TEST 5: Preprocessed OCR Test")
    print("-" * 50)
    
    # Set the tesseract path for Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        
        # Save preprocessed image for inspection
        cv2.imwrite("preprocessed_test.png", contrast)
        print("💾 Saved preprocessed image as 'preprocessed_test.png' for inspection")
        
        # OCR with config
        config = '--oem 3 --psm 6'
        text = pytesseract.image_to_string(contrast, config=config)
        
        print(f"📄 Preprocessed OCR text ({len(text)} characters):")
        print(f"'{text[:200]}{'...' if len(text) > 200 else ''}'")
        
        if len(text.strip()) > 10:
            print("✅ Preprocessed OCR successful")
            return True, text
        else:
            print("⚠️  Still getting minimal text - image quality might be poor")
            return False, text
            
    except Exception as e:
        print(f"❌ Preprocessed OCR failed: {e}")
        return False, ""

def test_ingredient_extraction(text):
    """Test 6: Ingredient list extraction"""
    print(f"\n🧪 TEST 6: Ingredient Extraction Test")
    print("-" * 50)
    
    if not text or len(text.strip()) < 10:
        print("❌ No text to extract ingredients from")
        return False
    
    try:
        import re
        
        # Look for ingredients section
        patterns = [
            r'ingredients?[:\-\s]*(.*?)(?=directions|instructions|warning|caution|$)',
            r'inci[:\-\s]*(.*?)(?=directions|instructions|warning|caution|$)',
        ]
        
        ingredients_found = False
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ingredients_text = match.group(1)
                ingredients = [ing.strip() for ing in ingredients_text.split(',')]
                ingredients = [ing for ing in ingredients if len(ing) > 2]
                
                if ingredients:
                    print(f"✅ Found {len(ingredients)} ingredients using pattern")
                    print("🧪 First few ingredients:")
                    for i, ing in enumerate(ingredients[:5]):
                        print(f"   {i+1}. {ing}")
                    ingredients_found = True
                    break
        
        if not ingredients_found:
            # Try simple comma splitting
            ingredients = [ing.strip() for ing in text.split(',')]
            ingredients = [ing for ing in ingredients if len(ing) > 2]
            
            if len(ingredients) > 1:
                print(f"✅ Found {len(ingredients)} potential ingredients (simple split)")
                print("🧪 First few items:")
                for i, ing in enumerate(ingredients[:5]):
                    print(f"   {i+1}. {ing}")
                ingredients_found = True
        
        return ingredients_found
        
    except Exception as e:
        print(f"❌ Ingredient extraction failed: {e}")
        return False

def run_complete_ocr_test(image_path=None):
    """Run all OCR tests"""
    print("🚀 COMPLETE OCR TESTING SUITE")
    print("=" * 60)
    
    # Test results
    results = {}
    
    # Test 1: Tesseract installation
    results['tesseract'] = test_tesseract_installation()
    
    # Test 2: Simple text recognition
    results['simple_text'] = test_simple_text_recognition()
    
    # If no image provided, ask for one
    if not image_path:
        print(f"\n📸 For the remaining tests, we need a test image.")
        print("Please save a cosmetic product ingredient label image in:")
        print("assets/test_images/your_image.jpg")
        
        # Try to find images automatically
        if os.path.exists("assets/test_images"):
            images = [f for f in os.listdir("assets/test_images") 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                image_path = os.path.join("assets/test_images", images[0])
                print(f"🎯 Found test image: {image_path}")
            else:
                print("❌ No images found in assets/test_images/")
                return results
        else:
            print("❌ assets/test_images/ directory not found")
            return results
    
    if image_path:
        # Test 3: Image loading
        results['image_loading'] = test_image_loading(image_path)
        
        if results['image_loading']:
            # Test 4: Raw OCR
            results['raw_ocr'], raw_text = test_raw_ocr(image_path)
            
            # Test 5: Preprocessed OCR
            results['preprocessed_ocr'], processed_text = test_preprocessed_ocr(image_path)
            
            # Test 6: Ingredient extraction
            best_text = processed_text if results['preprocessed_ocr'] else raw_text
            results['ingredient_extraction'] = test_ingredient_extraction(best_text)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test.replace('_', ' ').title():<25} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your OCR setup is working perfectly!")
    elif passed >= total * 0.8:
        print("✨ Most tests passed! OCR should work well.")
    else:
        print("⚠️  Several tests failed. Check the errors above for solutions.")
    
    return results

if __name__ == "__main__":
    # Run the complete test suite
    # You can specify an image path or let it auto-detect
    results = run_complete_ocr_test()
    
    # Keep window open on Windows
    input("\nPress Enter to close...")