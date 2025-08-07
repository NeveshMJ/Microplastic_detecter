# camera_scanner.py
"""
Live camera microplastic scanner - Capture and scan products in real-time
"""

import cv2
import pytesseract
import os
import json
from PIL import Image
import time
from datetime import datetime

class CameraMicroplasticScanner:
    def __init__(self):
        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Load microplastic database
        self.load_microplastic_database()
        
        # Create captures directory
        if not os.path.exists("captures"):
            os.makedirs("captures")
    
    def load_microplastic_database(self):
        """Load microplastic ingredients database"""
        # Simplified database for quick testing
        self.microplastic_ingredients = {
            'high_risk': [
                'polydimethylsiloxane', 'dimethicone', 'polyethylene', 'polypropylene',
                'polystyrene', 'polyurethane', 'polyacrylate', 'petrolatum', 
                'paraffinum liquidum', 'mineral oil', 'polytetrafluoroethylene',
                'polytrimethylsiloxysilicate', 'polyvinyl chloride'
            ],
            'medium_risk': [
                'polyvinyl alcohol', 'polyethylene glycol', 'polylactic acid',
                'isopropyl myristate', 'cetearyl alcohol'
            ],
            'patterns': [
                'poly', 'dimethicone', 'siloxane', 'acrylate', 'styrene'
            ]
        }
    
    def preprocess_image(self, frame):
        """Preprocess image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        
        # Resize if too small
        height, width = contrast.shape
        if width < 800:
            scale_factor = 800 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            contrast = cv2.resize(contrast, (new_width, new_height))
        
        return contrast
    
    def extract_text_from_frame(self, frame):
        """Extract text from camera frame"""
        try:
            # Preprocess
            processed = self.preprocess_image(frame)
            
            # OCR configuration for ingredient labels
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),-.: '
            
            # Extract text
            text = pytesseract.image_to_string(processed, config=config)
            
            # Clean text
            text = text.lower().strip()
            
            return text if len(text) > 10 else ""
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def find_microplastics_in_text(self, text):
        """Find microplastics in extracted text"""
        found_microplastics = {'high_risk': [], 'medium_risk': [], 'patterns': []}
        
        if not text:
            return found_microplastics
        
        # Check high risk ingredients
        for ingredient in self.microplastic_ingredients['high_risk']:
            if ingredient.lower() in text.lower():
                found_microplastics['high_risk'].append(ingredient)
        
        # Check medium risk ingredients
        for ingredient in self.microplastic_ingredients['medium_risk']:
            if ingredient.lower() in text.lower():
                found_microplastics['medium_risk'].append(ingredient)
        
        # Check patterns
        for pattern in self.microplastic_ingredients['patterns']:
            if pattern.lower() in text.lower():
                # Find the actual ingredient containing the pattern
                words = text.split()
                for word in words:
                    if pattern.lower() in word.lower() and len(word) > 4:
                        found_microplastics['patterns'].append(word)
        
        return found_microplastics
    
    def assess_safety(self, microplastics_found):
        """Assess product safety"""
        high_risk_count = len(microplastics_found['high_risk'])
        medium_risk_count = len(microplastics_found['medium_risk'])
        pattern_count = len(microplastics_found['patterns'])
        
        if high_risk_count > 0:
            return {
                'level': 'HIGH RISK',
                'color': (0, 0, 255),  # Red
                'message': f'⚠️ {high_risk_count} HARMFUL microplastics detected!'
            }
        elif medium_risk_count > 0 or pattern_count > 0:
            return {
                'level': 'MODERATE RISK', 
                'color': (0, 165, 255),  # Orange
                'message': f'⚠️ {medium_risk_count + pattern_count} questionable ingredients'
            }
        else:
            return {
                'level': 'SAFER CHOICE',
                'color': (0, 255, 0),  # Green
                'message': '✅ No obvious microplastics detected'
            }
    
    def draw_results_on_frame(self, frame, text, microplastics, safety):
        """Draw scan results on the camera frame"""
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, (10, 10), (width-10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw safety status
        cv2.putText(frame, safety['level'], (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, safety['color'], 2)
        
        cv2.putText(frame, safety['message'], (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw detected microplastics
        y_pos = 110
        if microplastics['high_risk']:
            cv2.putText(frame, f"High Risk: {', '.join(microplastics['high_risk'][:2])}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_pos += 25
        
        if microplastics['medium_risk']:
            cv2.putText(frame, f"Medium Risk: {', '.join(microplastics['medium_risk'][:2])}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            y_pos += 25
        
        # Draw instructions
        cv2.putText(frame, "SPACE: Capture & Save | 'q': Quit | 's': Detailed Scan", 
                   (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_capture(self, frame, text, microplastics, safety):
        """Save captured frame and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save image
        image_path = f"captures/scan_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        
        # Save results
        results = {
            'timestamp': timestamp,
            'image_path': image_path,
            'extracted_text': text,
            'microplastics_found': microplastics,
            'safety_assessment': safety,
            'ingredients_list': [ing.strip() for ing in text.split(',') if len(ing.strip()) > 2]
        }
        
        results_path = f"captures/results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Saved: {image_path} and {results_path}")
        return results
    
    def detailed_scan_report(self, results):
        """Show detailed scan report"""
        print("\n" + "="*60)
        print("🔬 DETAILED MICROPLASTIC SCAN REPORT")
        print("="*60)
        print(f"📅 Timestamp: {results['timestamp']}")
        print(f"⚖️  Safety Level: {results['safety_assessment']['level']}")
        print(f"💬 Assessment: {results['safety_assessment']['message']}")
        
        microplastics = results['microplastics_found']
        
        if microplastics['high_risk']:
            print(f"\n🔴 HIGH RISK Microplastics ({len(microplastics['high_risk'])}):")
            for ingredient in microplastics['high_risk']:
                print(f"  • {ingredient.title()}")
        
        if microplastics['medium_risk']:
            print(f"\n🟠 MEDIUM RISK Ingredients ({len(microplastics['medium_risk'])}):")
            for ingredient in microplastics['medium_risk']:
                print(f"  • {ingredient.title()}")
        
        if microplastics['patterns']:
            print(f"\n🔍 SUSPECTED Microplastics ({len(microplastics['patterns'])}):")
            for ingredient in microplastics['patterns']:
                print(f"  • {ingredient.title()}")
        
        print(f"\n🧪 Total Ingredients Found: {len(results['ingredients_list'])}")
        print(f"📄 Raw Text Length: {len(results['extracted_text'])} characters")
        
        print("\n💡 RECOMMENDATIONS:")
        if microplastics['high_risk']:
            print("❌ NOT RECOMMENDED - Contains harmful microplastics")
            print("🌍 These ingredients contribute to ocean pollution")
            print("🏥 May cause skin irritation and health concerns")
        elif microplastics['medium_risk'] or microplastics['patterns']:
            print("⚠️ USE WITH CAUTION - Contains questionable ingredients")
            print("🔍 Consider looking for alternatives")
        else:
            print("✅ BETTER CHOICE - No obvious harmful microplastics detected")
            print("🌿 This product appears safer for you and the environment")
        
        print("="*60)
    
    def run_live_scanner(self):
        """Run the live camera scanner"""
        print("🚀 Starting Live Microplastic Scanner...")
        print("📱 Point camera at ingredient labels")
        print("⌨️  Controls: SPACE=Capture, 's'=Detailed scan, 'q'=Quit")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        last_scan_time = 0
        current_text = ""
        current_microplastics = {'high_risk': [], 'medium_risk': [], 'patterns': []}
        current_safety = {'level': 'SCANNING...', 'color': (255, 255, 255), 'message': 'Point at ingredient label'}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Scan every 2 seconds to avoid overprocessing
            current_time = time.time()
            if current_time - last_scan_time > 2:
                current_text = self.extract_text_from_frame(frame)
                if current_text:
                    print(f"\n📄 EXTRACTED TEXT ({len(current_text)} chars):")
                    print("-" * 50)
                    print(f"{current_text}")
                    print("-" * 50)
                    current_microplastics = self.find_microplastics_in_text(current_text)
                    current_safety = self.assess_safety(current_microplastics)
                    print(f"🔍 Analysis: {current_safety['level']} - {current_safety['message']}")
                last_scan_time = current_time
            
            # Draw results on frame
            display_frame = self.draw_results_on_frame(
                frame.copy(), current_text, current_microplastics, current_safety
            )
            
            # Show frame
            cv2.imshow('Microplastic Scanner - Live Feed', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar
                if current_text:
                    results = self.save_capture(frame, current_text, current_microplastics, current_safety)
                    print(f"📸 Captured! Safety Level: {current_safety['level']}")
                else:
                    print("⚠️  No text detected - move closer to ingredient label")
            elif key == ord('s'):  # Detailed scan
                if current_text:
                    results = self.save_capture(frame, current_text, current_microplastics, current_safety)
                    self.detailed_scan_report(results)
                else:
                    print("⚠️  No text detected for detailed scan")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Scanner stopped. Check 'captures/' folder for saved scans!")

def main():
    scanner = CameraMicroplasticScanner()
    scanner.run_live_scanner()

if __name__ == "__main__":
    main()