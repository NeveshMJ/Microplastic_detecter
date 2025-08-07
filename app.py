
from flask import Flask, render_template, request, jsonify, Response, send_file
import cv2
import os
import sys
import json
import easyocr
import re
import numpy as np
from typing import Dict, List, Tuple
from rapidfuzz import fuzz
import sqlite3
from datetime import datetime
import time
import threading
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

class WebCameraCaptureAnalyzer:
    def __init__(self):
        print("🚀 Initializing Web Camera Capture Analyzer...")
        
        # Initialize OCR
        print("🔄 Loading OCR engine...")
        self.reader = easyocr.Reader(['en'])
        print("✅ OCR ready!")
        
        # Load microplastic database
        self.microplastic_db = self.load_microplastic_database()
        print(f"✅ Loaded {len(self.microplastic_db)} microplastic ingredients")
        
        # Initialize database
        self.init_database()
        
        # Create captures directory
        if not os.path.exists("static/captures"):
            os.makedirs("static/captures")
        
        # Camera variables
        self.camera = None
        self.camera_active = False
        
        print("✅ All components loaded!")
    
    def load_microplastic_database(self) -> Dict:
        """Load the microplastic ingredients database"""
        microplastic_ingredients = {
            # Silicones (very common in cosmetics)
            "Polydimethylsiloxane": {
                "aliases": ["Dimethicone", "PDMS", "Silicone", "Dimethyl siloxane"],
                "risk_level": "RED",
                "category": "Silicone polymer",
                "health_concerns": [
                    "Environmental pollution",
                    "Bioaccumulation in marine life",
                    "Non-biodegradable",
                    "Microplastic shedding"
                ],
                "common_in": ["Moisturizers", "Sunscreens", "Foundations", "Hair products"]
            },
            "Polytrimethylsiloxysilicate": {
                "aliases": ["Silicone resin", "TMS", "Trimethylsiloxysilicate"],
                "risk_level": "RED",
                "category": "Silicone resin",
                "health_concerns": [
                    "Environmental persistence",
                    "Microplastic formation"
                ],
                "common_in": ["Long-wear makeup", "Sunscreens"]
            },
            "Cyclopentasiloxane": {
                "aliases": ["D5", "Cyclomethicone"],
                "risk_level": "RED",
                "category": "Cyclic silicone",
                "health_concerns": [
                    "Environmental accumulation",
                    "Potential endocrine disruptor"
                ],
                "common_in": ["Deodorants", "Hair products", "Skin care"]
            },
            
            # Acrylates (common in makeup and nail products)
            "Polyacrylate": {
                "aliases": ["Acrylate polymer", "Poly(acrylic acid)", "Acrylic polymer"],
                "risk_level": "RED",
                "category": "Acrylic polymer",
                "health_concerns": [
                    "Microplastic shedding",
                    "Environmental accumulation"
                ],
                "common_in": ["Nail polish", "Hair gel", "Foundation"]
            },
            "Polybutyl Acrylate": {
                "aliases": ["PBA", "Butyl acrylate polymer"],
                "risk_level": "RED",
                "category": "Acrylic polymer",
                "health_concerns": [
                    "Microplastic formation",
                    "Skin irritation potential"
                ],
                "common_in": ["Makeup", "Nail products"]
            },
            
            # Methacrylates
            "Polymethyl Methacrylate": {
                "aliases": ["PMMA", "Acrylic glass", "Plexiglass", "Methyl methacrylate polymer"],
                "risk_level": "RED",
                "category": "Methacrylate polymer",
                "health_concerns": [
                    "Microplastic formation",
                    "Environmental persistence"
                ],
                "common_in": ["Nail polish", "Makeup", "Exfoliants"]
            },
            
            # Polyethylene (microbeads)
            "Polyethylene": {
                "aliases": ["PE", "Polyethylene microbeads", "Polyethylene glycol"],
                "risk_level": "RED",
                "category": "Polyolefin",
                "health_concerns": [
                    "Marine pollution",
                    "Bioaccumulation",
                    "Banned in many countries"
                ],
                "common_in": ["Exfoliating scrubs", "Toothpaste"]
            },
            
            # Nylon
            "Polycaprolactam": {
                "aliases": ["Nylon 6", "PA6", "Nylon"],
                "risk_level": "RED",
                "category": "Polyamide",
                "health_concerns": [
                    "Microplastic formation",
                    "Environmental accumulation"
                ],
                "common_in": ["Makeup brushes", "Some cosmetics"]
            },
            
            # Teflon
            "Polytetrafluoroethylene": {
                "aliases": ["PTFE", "Teflon"],
                "risk_level": "RED",
                "category": "Fluoropolymer",
                "health_concerns": [
                    "Environmental persistence",
                    "Bioaccumulation",
                    "PFAS contamination"
                ],
                "common_in": ["Waterproof makeup", "Some foundations"]
            }
        }
        return microplastic_ingredients
    
    def init_database(self):
        """Initialize SQLite database for storing scan results"""
        conn = sqlite3.connect('cosmetics_scanner.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT,
                brand TEXT,
                category TEXT,
                safety_score INTEGER,
                safety_level TEXT,
                total_ingredients INTEGER,
                microplastic_count INTEGER,
                scan_date TIMESTAMP,
                image_path TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS product_ingredients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                ingredient_name TEXT,
                is_microplastic BOOLEAN,
                risk_level TEXT,
                confidence_score REAL,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def find_working_camera(self):
        """Find working camera with multiple backends"""
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for camera_idx in range(5):
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(camera_idx, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            return cap, camera_idx
                    cap.release()
                except Exception:
                    continue
        return None, -1
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from ingredient list image using OCR"""
        try:
            result = self.reader.readtext(image_path)
            text_parts = []
            for bbox, text, confidence in result:
                if confidence > 0.3:
                    text_parts.append(text)
            return ' '.join(text_parts)
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return ""
    
    def validate_ingredient_content(self, text: str) -> bool:
        """Validate if the extracted text contains ingredient-like content"""
        if not text or len(text.strip()) < 10:
            return False
        
        text_lower = text.lower()
        
        # Look for ingredient indicators
        ingredient_keywords = [
            'ingredients', 'inci', 'composition', 'contains', 'formula',
            'aqua', 'water', 'glycerin', 'alcohol', 'parfum', 'fragrance',
            'sodium', 'acid', 'oil', 'extract', 'oxide', 'sulfate',
            'paraben', 'silicone', 'dimethicone', 'cetyl', 'stearyl'
        ]
        
        # Check for common cosmetic ingredient patterns
        cosmetic_patterns = [
            r'\b\w*acid\b',  # Various acids
            r'\b\w*yl\b',    # Common endings like cetyl, stearyl
            r'\b\w*ate\b',   # Sulfates, acetates, etc.
            r'\b\w*ol\b',    # Alcohols
            r'\b\w*ene\b',   # Glycol compounds
            r'aqua|water',   # Water/aqua
            r'ci \d+',       # Color index
            r'fd&c',         # Food coloring
        ]
        
        # Count matches
        keyword_matches = sum(1 for keyword in ingredient_keywords if keyword in text_lower)
        pattern_matches = sum(1 for pattern in cosmetic_patterns if re.search(pattern, text_lower))
        
        # Check for ingredient list structure (commas, chemical names)
        comma_count = text.count(',')
        has_chemical_structure = bool(re.search(r'\b[a-z]{3,}\s*[a-z]{3,}', text_lower))
        
        # Validation criteria
        min_matches_needed = 2
        min_comma_count = 2
        
        is_valid = (
            (keyword_matches >= min_matches_needed or pattern_matches >= min_matches_needed) and
            (comma_count >= min_comma_count or has_chemical_structure) and
            len(text.split()) >= 5  # At least 5 words
        )
        
        return is_valid
    
    def parse_ingredients(self, text: str) -> List[str]:
        """Parse and clean ingredient list from extracted text"""
        if not text:
            return []
        
        # Clean the text
        text = re.sub(r'[^\w\s,.\-\(\)\/:]', '', text)
        
        # Split by common separators
        ingredients = re.split(r'[,.\n:]', text)
        
        # Clean each ingredient
        cleaned_ingredients = []
        for ingredient in ingredients:
            ingredient = ingredient.strip()
            ingredient = re.sub(r'\([^)]*\)', '', ingredient).strip()
            
            if len(ingredient) < 3:
                continue
                
            skip_words = ['ingredients', 'contains', 'may contain', 'and', 'or', 'the', 'in', 'with', 'inc', 'ltd']
            if ingredient.lower() not in skip_words:
                cleaned_ingredients.append(ingredient.title())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ingredients = []
        for ingredient in cleaned_ingredients:
            if ingredient.lower() not in seen:
                seen.add(ingredient.lower())
                unique_ingredients.append(ingredient)
        
        return unique_ingredients
    
    def fuzzy_match_ingredient(self, ingredient: str) -> Tuple[bool, str, int]:
        """Use fuzzy matching to identify microplastic ingredients"""
        best_match = None
        best_score = 0
        
        ingredient_lower = ingredient.lower()
        
        for mp_ingredient, data in self.microplastic_db.items():
            # Check exact match
            if ingredient_lower == mp_ingredient.lower():
                return True, mp_ingredient, 100
            
            # Check aliases
            for alias in data['aliases']:
                if ingredient_lower == alias.lower():
                    return True, mp_ingredient, 100
            
            # Fuzzy matching against main ingredient name
            score = fuzz.partial_ratio(ingredient_lower, mp_ingredient.lower())
            if score > best_score:
                best_score = score
                best_match = mp_ingredient
            
            # Check aliases with fuzzy matching
            for alias in data['aliases']:
                alias_score = fuzz.partial_ratio(ingredient_lower, alias.lower())
                if alias_score > best_score:
                    best_score = alias_score
                    best_match = mp_ingredient
        
        if best_score >= 75:
            return True, best_match, best_score
        
        return False, None, 0
    
    def analyze_captured_image(self, image_path, product_name="", brand=""):
        """Run full analysis on captured image"""
        # Extract text using OCR
        extracted_text = self.extract_text_from_image(image_path)
        
        if not extracted_text:
            return {"error": "Could not extract text from image. Try better lighting or clearer image."}
        
        # Validate if the extracted text contains ingredient-like content
        if not self.validate_ingredient_content(extracted_text):
            return {"error": "Invalid input: This image does not appear to contain a cosmetic ingredient list. Please upload an image showing the ingredients section of a cosmetic product."}
        
        # Extract ingredients
        ingredients_list = self.parse_ingredients(extracted_text)
        
        if not ingredients_list:
            return {"error": "Could not identify ingredients from extracted text. Please ensure ingredient list is visible and clear."}
        
        # Detect microplastics
        microplastic_ingredients = []
        safe_ingredients = []
        
        for ingredient in ingredients_list:
            is_microplastic, matched_ingredient, confidence = self.fuzzy_match_ingredient(ingredient)
            
            if is_microplastic:
                mp_data = self.microplastic_db[matched_ingredient]
                microplastic_ingredients.append({
                    'ingredient': ingredient,
                    'matched_as': matched_ingredient,
                    'confidence': confidence / 100,
                    'risk_level': mp_data['risk_level'],
                    'health_concerns': mp_data['health_concerns'],
                    'category': mp_data['category']
                })
            else:
                safe_ingredients.append(ingredient)
        
        # Assess safety
        microplastic_count = len(microplastic_ingredients)
        total_count = len(ingredients_list)
        
        if total_count == 0:
            safety_score = 0
        else:
            penalty_per_microplastic = 25
            safety_score = max(0, 100 - (microplastic_count * penalty_per_microplastic))
        
        # Determine risk level
        if safety_score >= 80:
            risk_level = "GREEN"
            safety_rating = "SAFE"
            warnings = "No harmful microplastics detected"
        elif safety_score >= 50:
            risk_level = "YELLOW"
            safety_rating = "CAUTION"
            warnings = f"Found {microplastic_count} microplastic ingredient(s)"
        else:
            risk_level = "RED"
            safety_rating = "NOT RECOMMENDED"
            warnings = f"Multiple harmful microplastics detected ({microplastic_count})"
        
        # Prepare results
        results = {
            'product_name': product_name,
            'brand': brand,
            'image_path': image_path,
            'ingredients': ', '.join(ingredients_list),
            'microplastics_found': {
                'red': microplastic_ingredients,
                'orange': [],
                'green': [{'ingredient': ing} for ing in safe_ingredients]
            },
            'safety_rating': safety_rating,
            'risk_level': risk_level,
            'warnings': warnings,
            'extracted_text': extracted_text,
            'total_ingredients': len(ingredients_list),
            'safety_score': safety_score
        }
        
        # Save to database
        self.save_to_database(results)
        
        return results
    
    def save_to_database(self, results):
        """Save analysis results to database"""
        try:
            conn = sqlite3.connect('cosmetics_scanner.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO products (product_name, brand, safety_score, safety_level, 
                                    total_ingredients, microplastic_count, scan_date, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                results['product_name'],
                results['brand'],
                results['safety_score'],
                results['risk_level'],
                results['total_ingredients'],
                len(results['microplastics_found']['red']),
                datetime.now(),
                results['image_path']
            ))
            
            product_id = cursor.lastrowid
            
            # Insert microplastic ingredients
            for mp in results['microplastics_found']['red']:
                cursor.execute('''
                    INSERT INTO product_ingredients (product_id, ingredient_name, 
                                                   is_microplastic, risk_level, confidence_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    product_id,
                    mp['ingredient'],
                    True,
                    mp['risk_level'],
                    mp['confidence']
                ))
            
            # Insert safe ingredients
            for safe in results['microplastics_found']['green']:
                cursor.execute('''
                    INSERT INTO product_ingredients (product_id, ingredient_name, 
                                                   is_microplastic, risk_level)
                    VALUES (?, ?, ?, ?)
                ''', (
                    product_id,
                    safe['ingredient'],
                    False,
                    'GREEN'
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
    
    def get_product_history(self):
        """Get analysis history from database"""
        try:
            conn = sqlite3.connect('cosmetics_scanner.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, product_name, brand, safety_score, safety_level, 
                       total_ingredients, microplastic_count, scan_date
                FROM products 
                ORDER BY scan_date DESC 
                LIMIT 20
            ''')
            
            history = cursor.fetchall()
            conn.close()
            return history
            
        except Exception as e:
            print(f"❌ Error getting history: {e}")
            return []

# Initialize the analyzer
analyzer = WebCameraCaptureAnalyzer()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/camera_feed')
def camera_feed():
    """Video streaming route"""
    def generate():
        cap, camera_idx = analyzer.find_working_camera()
        if cap is None:
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Add overlay
                h, w = frame.shape[:2]
                if h > 80 and w > 300:
                    cv2.putText(frame, "Position product in center - Click Capture when ready", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Add center crosshair
                    center_x, center_y = w // 2, h // 2
                    cv2.line(frame, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 0), 2)
                    cv2.line(frame, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 0), 2)
                
                # Convert frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            cap.release()
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Capture image from camera"""
    cap, camera_idx = analyzer.find_working_camera()
    if cap is None:
        return jsonify({'error': 'No camera found'})
    
    try:
        ret, frame = cap.read()
        if ret and frame is not None:
            # Flip frame for consistency with preview
            frame = cv2.flip(frame, 1)
            
            # Basic image quality check
            if frame.size == 0:
                return jsonify({'error': 'Captured image is empty'})
            
            # Check if image is too dark or too bright
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 30:
                return jsonify({
                    'error': 'Image too dark - please ensure good lighting',
                    'warning': True
                })
            elif mean_brightness > 220:
                return jsonify({
                    'error': 'Image too bright - adjust lighting or camera position',
                    'warning': True
                })
            
            # Save captured image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"static/captures/captured_{timestamp}.jpg"
            cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return jsonify({
                'success': True,
                'image_path': image_path,
                'message': 'Image captured successfully!'
            })
        else:
            return jsonify({'error': 'Failed to capture image'})
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        cap.release()

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Analyze captured image"""
    data = request.get_json()
    image_path = data.get('image_path')
    product_name = data.get('product_name', '')
    brand = data.get('brand', '')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Image file not found'})
    
    try:
        results = analyzer.analyze_captured_image(image_path, product_name, brand)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/history')
def get_history():
    """Get scan history"""
    history = analyzer.get_product_history()
    return jsonify(history)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Upload image file for analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uploaded_{timestamp}_{file.filename}"
        image_path = f"static/captures/{filename}"
        file.save(image_path)
        
        return jsonify({
            'success': True,
            'image_path': image_path,
            'message': 'File uploaded successfully!'
        })
    
    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
