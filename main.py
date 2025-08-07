import os
from utils.ocr_processor import OCRProcessor
from models.ingredient_detector import IngredientDetector
from models.database_manager import DatabaseManager

class MicroplasticDetector:
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.ingredient_detector = IngredientDetector()
        self.db_manager = DatabaseManager()
    
    def scan_product(self, image_path, product_name="", brand=""):
        """Main function to scan a product image"""
        print(f"🔍 Scanning product: {product_name or 'Unknown'}")
        
        # Step 1: Extract text using OCR
        print("📝 Extracting text from image...")
        extracted_text = self.ocr_processor.extract_text_from_image(image_path)
        
        if not extracted_text:
            return {"error": "Could not extract text from image"}
        
        print(f"✅ Extracted text: {extracted_text[:100]}...")
        
        # Step 2: Extract ingredients
        print("🧪 Identifying ingredients...")
        ingredients_list = self.ocr_processor.extract_ingredients_list(extracted_text)
        
        if not ingredients_list:
            return {"error": "Could not identify ingredients list"}
        
        print(f"✅ Found {len(ingredients_list)} ingredients")
        
        # Step 3: Detect microplastics
        print("🔬 Analyzing for microplastics...")
        detected_microplastics = self.ingredient_detector.find_microplastics(ingredients_list)
        
        # Step 4: Assess safety
        safety_assessment = self.ingredient_detector.assess_safety(detected_microplastics)
        
        # Step 5: Prepare results
        results = {
            'product_name': product_name,
            'brand': brand,
            'ingredients': ', '.join(ingredients_list),
            'microplastics_found': detected_microplastics,
            'safety_rating': safety_assessment['safety_rating'],
            'risk_level': safety_assessment['risk_level'],
            'warnings': safety_assessment['warnings']
        }
        
        # Step 6: Save to database
        self.db_manager.save_product(results)
        
        return results
    
    def display_results(self, results):
        """Display scan results in a formatted way"""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("\n" + "="*50)
        print("📊 MICROPLASTIC DETECTION RESULTS")
        print("="*50)
        
        print(f"🏷️  Product: {results['product_name'] or 'Unknown'}")
        print(f"🏢 Brand: {results['brand'] or 'Unknown'}")
        print(f"⚖️  Safety Rating: {results['safety_rating']}")
        print(f"⚠️  Risk Level: {results['risk_level']}")
        print(f"💡 Warnings: {results['warnings']}")
        
        print(f"\n🧪 Ingredients Found: {len(results['ingredients'].split(','))}")
        
        microplastics = results['microplastics_found']
        
        if microplastics['red']:
            print(f"\n🔴 HARMFUL Microplastics ({len(microplastics['red'])}):")
            for item in microplastics['red']:
                print(f"  • {item['ingredient']} (matched as: {item['matched_as']})")
        
        if microplastics['orange']:
            print(f"\n🟠 QUESTIONABLE Ingredients ({len(microplastics['orange'])}):")
            for item in microplastics['orange']:
                print(f"  • {item['ingredient']} (matched as: {item['matched_as']})")
        
        if microplastics['green']:
            print(f"\n🟢 SAFER Alternatives Found ({len(microplastics['green'])}):")
            for item in microplastics['green']:
                print(f"  • {item['ingredient']}")
        
        print("="*50)

def main():
    detector = MicroplasticDetector()
    
    while True:
        print("\n🌟 MICROPLASTIC DETECTOR")
        print("1. Scan product image")
        print("2. View scan history")
        print("3. Exit")
        
        choice = input("\nChoose option (1-3): ")
        
        if choice == '1':
            image_path = input("📸 Enter image path: ").strip()
            
            if not os.path.exists(image_path):
                print("❌ Image file not found!")
                continue
            
            product_name = input("🏷️  Product name (optional): ").strip()
            brand = input("🏢 Brand name (optional): ").strip()
            
            print("\n🚀 Starting scan...")
            results = detector.scan_product(image_path, product_name, brand)
            detector.display_results(results)
        
        elif choice == '2':
            history = detector.db_manager.get_product_history()
            if not history:
                print("📭 No scan history found!")
            else:
                print(f"\n📚 Found {len(history)} scanned products:")
                for i, product in enumerate(history[:10]):  # Show last 10
                    print(f"{i+1}. {product[1] or 'Unknown'} - {product[5]} ({product[6]})")
        
        elif choice == '3':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()