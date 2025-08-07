from utils.ocr_processor import OCRProcessor
from models.ingredient_detector import IngredientDetector

# Test the system
ocr = OCRProcessor()
detector = IngredientDetector()

image_path = "assets/test_images/test1.jpg"

print("🔍 Extracting text...")
text = ocr.extract_text_from_image(image_path)
print(f"Text: {text[:100]}...")

print("\n🧪 Finding ingredients...")  
ingredients = ocr.extract_ingredients_list(text)
print(f"Ingredients found: {ingredients[:5]}...")

print("\n🔬 Checking for microplastics...")
microplastics = detector.find_microplastics(ingredients)
safety = detector.assess_safety(microplastics)

print(f"\nSafety Rating: {safety['safety_rating']}")
print(f"Risk Level: {safety['risk_level']}")
print(f"Warnings: {safety['warnings']}")

if microplastics['red']:
    print(f"\n🔴 Harmful ingredients found:")
    for item in microplastics['red']:
        print(f"  • {item['ingredient']}")