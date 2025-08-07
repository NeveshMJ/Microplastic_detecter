# camera_capture_analyzer_camera_fix.py
"""
Camera Capture Analyzer - Fixed for Windows Camera Issues
Handles camera stream and matrix errors
"""

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

class CameraCaptureAnalyzer:
    def __init__(self):
        print("🚀 Initializing Camera Capture Analyzer...")
        
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
        if not os.path.exists("captures"):
            os.makedirs("captures")
        
        # Check OpenCV GUI support
        self.gui_available = self.check_opencv_gui_support()
        
        print("✅ All components loaded!")
    
    def check_opencv_gui_support(self):
        """Check if OpenCV GUI functions are available"""
        try:
            # Try to create a simple test window
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.imshow("test", test_img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            print("✅ OpenCV GUI support available")
            return True
        except Exception as e:
            print("⚠️  OpenCV GUI not available - using alternative methods")
            return False
    
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
    
    def find_working_camera_advanced(self):
        """Enhanced camera detection with multiple backends and error handling"""
        print("🔍 Advanced camera search with Windows-specific fixes...")
        
        # List of backends to try (Windows-specific)
        backends = [
            cv2.CAP_DSHOW,    # DirectShow (Windows)
            cv2.CAP_MSMF,     # Microsoft Media Foundation
            cv2.CAP_ANY,      # Auto-detect
        ]
        
        for camera_idx in range(5):  # Try first 5 camera indices
            print(f"   Testing camera {camera_idx}...")
            
            for backend in backends:
                try:
                    backend_name = {
                        cv2.CAP_DSHOW: "DirectShow",
                        cv2.CAP_MSMF: "Media Foundation", 
                        cv2.CAP_ANY: "Auto"
                    }.get(backend, "Unknown")
                    
                    print(f"     Trying {backend_name} backend...")
                    
                    # Create capture with specific backend
                    cap = cv2.VideoCapture(camera_idx, backend)
                    
                    if not cap.isOpened():
                        cap.release()
                        continue
                    
                    # Configure camera with safe settings
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test frame capture with retry mechanism
                    success_count = 0
                    for attempt in range(10):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            # Validate frame dimensions
                            if len(frame.shape) == 3 and frame.shape[0] > 0 and frame.shape[1] > 0:
                                success_count += 1
                                if success_count >= 3:  # Need at least 3 successful reads
                                    height, width = frame.shape[:2]
                                    print(f"✅ Camera {camera_idx} working with {backend_name}")
                                    print(f"   Resolution: {width}x{height}")
                                    return cap, camera_idx, backend
                        time.sleep(0.1)  # Small delay between attempts
                    
                    cap.release()
                    
                except Exception as e:
                    print(f"     {backend_name} failed: {str(e)[:50]}...")
                    continue
        
        print("❌ No working camera found with any backend!")
        return None, -1, None
    
    def safe_destroy_windows(self):
        """Safely destroy OpenCV windows"""
        try:
            if self.gui_available:
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Process window events
        except Exception:
            pass
    
    def capture_with_robust_camera_handling(self, cap):
        """Robust camera capture with proper error handling"""
        print("\n📹 ENHANCED LIVE PREVIEW MODE")
        print("=" * 50)
        print("🎥 Starting camera with enhanced error handling...")
        
        # Camera warmup with validation
        print("🔄 Camera initialization and validation...")
        warmup_success = 0
        
        for i in range(30):  # Extended warmup
            try:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Validate frame structure
                    if len(frame.shape) == 3 and frame.shape[0] > 0 and frame.shape[1] > 0:
                        warmup_success += 1
                        if i % 10 == 0:
                            print(f"   Warming up... {i+1}/30 (✅ {warmup_success} good frames)")
                    else:
                        print(f"   Frame {i+1}: Invalid dimensions")
                else:
                    print(f"   Frame {i+1}: Read failed")
                    
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   Frame {i+1}: Error - {str(e)[:30]}...")
                continue
        
        if warmup_success < 10:
            print(f"❌ Camera unstable - only {warmup_success}/30 successful reads")
            return None
        
        print(f"✅ Camera ready! ({warmup_success}/30 successful frames)")
        
        # Create window safely
        window_name = "📹 Live Camera Preview"
        
        try:
            if self.gui_available:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 800, 600)
                print("✅ Preview window created")
            else:
                print("⚠️  GUI not available - using frame saving mode")
                return self.capture_with_frame_saving(cap)
        except Exception as e:
            print(f"⚠️  Window creation failed: {e}")
            return self.capture_with_frame_saving(cap)
        
        captured_frame = None
        frame_counter = 0
        last_good_frame = None
        
        print("\n🎮 CONTROLS:")
        print("   📸 Press SPACEBAR to capture")
        print("   🔄 Press 'r' to refresh")
        print("   🚪 Press 'q' or ESC to quit")
        print("   💾 Press 's' to save current frame")
        print("\n⌨️  Focus on the preview window and use keyboard controls")
        
        try:
            while True:
                try:
                    ret, frame = cap.read()
                    
                    if not ret or frame is None or frame.size == 0:
                        print("⚠️  Frame read failed - using last good frame")
                        if last_good_frame is not None:
                            frame = last_good_frame.copy()
                        else:
                            continue
                    else:
                        # Validate frame
                        if len(frame.shape) != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
                            print("⚠️  Invalid frame structure")
                            continue
                        
                        last_good_frame = frame.copy()
                        frame_counter += 1
                    
                    # Flip frame for mirror effect
                    display_frame = cv2.flip(frame, 1)
                    
                    # Add overlay with safe dimensions
                    h, w = display_frame.shape[:2]
                    if h > 80 and w > 300:  # Ensure frame is large enough
                        # Add status overlay
                        overlay = np.zeros((60, w, 3), dtype=np.uint8)
                        cv2.putText(overlay, f"Frame: {frame_counter} | SPACEBAR=Capture | Q=Quit", 
                                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(overlay, "Position your product in the center", 
                                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Blend overlay
                        display_frame[:60, :] = cv2.addWeighted(display_frame[:60, :], 0.7, overlay, 0.3, 0)
                        
                        # Add center crosshair
                        center_x, center_y = w // 2, h // 2
                        cv2.line(display_frame, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 0), 2)
                        cv2.line(display_frame, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 0), 2)
                    
                    # Display frame
                    cv2.imshow(window_name, display_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord(' '):  # Spacebar - capture
                        print("\n📸 CAPTURING IMAGE!")
                        captured_frame = last_good_frame.copy() if last_good_frame is not None else frame.copy()
                        
                        # Flash effect
                        try:
                            flash_frame = np.ones_like(display_frame) * 255
                            cv2.imshow(window_name, flash_frame)
                            cv2.waitKey(200)
                        except Exception:
                            pass
                        
                        print("✅ Image captured successfully!")
                        
                        # Confirmation
                        try:
                            preview_frame = cv2.flip(captured_frame, 1)
                            cv2.putText(preview_frame, "CAPTURED! Press ENTER to use or SPACEBAR to retake", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.imshow(window_name, preview_frame)
                            
                            while True:
                                confirm_key = cv2.waitKey(0) & 0xFF
                                if confirm_key == 13 or confirm_key == ord('\r'):  # Enter
                                    print("✅ Image confirmed for analysis!")
                                    self.safe_destroy_windows()
                                    return captured_frame
                                elif confirm_key == ord(' '):  # Spacebar
                                    print("🔄 Taking new capture...")
                                    break
                                elif confirm_key == ord('q') or confirm_key == 27:  # q or ESC
                                    print("❌ Capture cancelled")
                                    self.safe_destroy_windows()
                                    return None
                        except Exception:
                            # Fallback to console confirmation
                            while True:
                                try:
                                    confirm = input("✅ Use captured image? (y/n): ").strip().lower()
                                    if confirm in ['y', 'yes', '']:
                                        return captured_frame
                                    elif confirm in ['n', 'no']:
                                        break
                                    else:
                                        print("Please enter 'y' or 'n'")
                                except (EOFError, KeyboardInterrupt):
                                    return None
                    
                    elif key == ord('s'):  # Save current frame
                        if last_good_frame is not None:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = f"captures/frame_save_{timestamp}.jpg"
                            cv2.imwrite(save_path, last_good_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            print(f"💾 Frame saved: {save_path}")
                    
                    elif key == ord('r'):  # Refresh
                        print("🔄 Refreshing camera...")
                        frame_counter = 0
                        # Clear buffer
                        for _ in range(5):
                            cap.read()
                    
                    elif key == ord('q') or key == 27:  # Quit
                        print("❌ Preview cancelled")
                        break
                    
                    # Check if window is still open
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            print("❌ Preview window closed")
                            break
                    except Exception:
                        break
                        
                except Exception as frame_error:
                    print(f"⚠️  Frame processing error: {str(frame_error)[:50]}...")
                    time.sleep(0.1)
                    continue
        
        except Exception as e:
            print(f"❌ Live preview error: {e}")
            return self.capture_with_frame_saving(cap)
        
        finally:
            self.safe_destroy_windows()
        
        return captured_frame
    
    def capture_with_frame_saving(self, cap):
        """Alternative capture method that saves frames to disk"""
        print("\n📁 FRAME SAVING CAPTURE MODE")
        print("=" * 50)
        print("💾 Saving frames for external viewing...")
        
        frame_counter = 0
        
        while True:
            print("\n📋 FRAME SAVING OPTIONS:")
            print("1. 📸 Capture current frame")
            print("2. 📷 Save preview frame to check positioning")
            print("3. ⏱️  Auto-capture with countdown")
            print("4. ❌ Cancel")
            
            try:
                choice = input("Choose option (1-4): ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            
            if choice == '1':
                print("\n📸 Capturing frame...")
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        frame_counter += 1
                        capture_path = f"captures/capture_{frame_counter}.jpg"
                        cv2.imwrite(frame, capture_path, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        print(f"✅ Frame captured: {capture_path}")
                        print("👁️  Open this file to see what was captured")
                        
                        while True:
                            try:
                                confirm = input("✅ Use this image for analysis? (y/n): ").strip().lower()
                                if confirm in ['y', 'yes', '']:
                                    return frame
                                elif confirm in ['n', 'no']:
                                    break
                                else:
                                    print("Please enter 'y' or 'n'")
                            except (EOFError, KeyboardInterrupt):
                                return None
                    else:
                        print("❌ Failed to capture frame")
                except Exception as e:
                    print(f"❌ Capture error: {e}")
            
            elif choice == '2':
                print("\n📷 Saving preview frame...")
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        preview_path = f"captures/preview_{int(time.time())}.jpg"
                        cv2.imwrite(preview_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        print(f"📁 Preview saved: {preview_path}")
                        print("👁️  Check this file to see current camera view")
                    else:
                        print("❌ Failed to get preview frame")
                except Exception as e:
                    print(f"❌ Preview error: {e}")
            
            elif choice == '3':
                print("\n⏱️  AUTO-CAPTURE WITH COUNTDOWN")
                
                try:
                    countdown_time = int(input("⏰ Countdown seconds (default 5): ").strip() or "5")
                except ValueError:
                    countdown_time = 5
                except (EOFError, KeyboardInterrupt):
                    continue
                
                print(f"\n🎯 Auto-capturing in {countdown_time} seconds...")
                for i in range(countdown_time, 0, -1):
                    print(f"   {i}...")
                    time.sleep(1)
                
                print("📸 CAPTURING NOW!")
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        auto_path = f"captures/auto_capture_{int(time.time())}.jpg"
                        cv2.imwrite(auto_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        print(f"✅ Auto-capture complete: {auto_path}")
                        
                        while True:
                            try:
                                confirm = input("✅ Use this image for analysis? (y/n): ").strip().lower()
                                if confirm in ['y', 'yes', '']:
                                    return frame
                                elif confirm in ['n', 'no']:
                                    break
                                else:
                                    print("Please enter 'y' or 'n'")
                            except (EOFError, KeyboardInterrupt):
                                return None
                    else:
                        print("❌ Auto-capture failed")
                except Exception as e:
                    print(f"❌ Auto-capture error: {e}")
            
            elif choice == '4':
                return None
            
            else:
                print("❌ Invalid choice!")
    
    def capture_image_from_camera(self):
        """Main camera capture with enhanced error handling"""
        print("\n📸 STARTING ENHANCED CAMERA CAPTURE")
        print("=" * 60)
        
        # Find camera with advanced detection
        cap, camera_idx, backend = self.find_working_camera_advanced()
        if cap is None:
            print("❌ No working camera found!")
            print("\n💡 TROUBLESHOOTING:")
            print("   - Check if camera is connected")
            print("   - Close other apps using the camera")
            print("   - Try different USB ports")
            print("   - Restart the application")
            return None
        
        backend_name = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "Media Foundation",
            cv2.CAP_ANY: "Auto"
        }.get(backend, "Unknown")
        
        print(f"📷 Using camera {camera_idx} with {backend_name} backend")
        
        captured_image = None
        
        try:
            if self.gui_available:
                print("✅ Starting enhanced live preview...")
                captured_image = self.capture_with_robust_camera_handling(cap)
            else:
                print("⚠️  GUI not available - using frame saving mode...")
                captured_image = self.capture_with_frame_saving(cap)
                
        except Exception as e:
            print(f"❌ Capture error: {e}")
            print("🔄 Trying fallback method...")
            try:
                captured_image = self.capture_with_frame_saving(cap)
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
            
        finally:
            if cap:
                cap.release()
            self.safe_destroy_windows()
        
        return captured_image
    
    def save_captured_image(self, image):
        """Save captured image with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"captures/captured_{timestamp}.jpg"
        
        # Save with high quality for better OCR
        cv2.imwrite(image_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"💾 Image saved: {image_path}")
        
        return image_path
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from ingredient list image using OCR"""
        try:
            print(f"🔍 Processing image with OCR...")
            
            # Use EasyOCR to extract text
            result = self.reader.readtext(image_path)
            
            # Combine all detected text with confidence scores
            text_parts = []
            total_confidence = 0
            print(f"📝 OCR found {len(result)} text regions:")
            for i, (bbox, text, confidence) in enumerate(result):
                if confidence > 0.3:  # Lower threshold for better coverage
                    text_parts.append(text)
                    total_confidence += confidence
                    print(f"  {i+1}. '{text}' (confidence: {confidence:.2f})")
            
            if text_parts:
                avg_confidence = total_confidence / len(text_parts)
                print(f"✅ Average OCR confidence: {avg_confidence:.2f}")
            
            extracted_text = ' '.join(text_parts)
            print(f"📄 Combined extracted text: {extracted_text[:200]}...")
            return extracted_text
            
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return ""
    
    def parse_ingredients(self, text: str) -> List[str]:
        """Parse and clean ingredient list from extracted text"""
        if not text:
            return []
            
        print(f"🔄 Parsing ingredients...")
        
        # Clean the text
        text = re.sub(r'[^\w\s,.\-\(\)\/:]', '', text)
        
        # Split by common separators
        ingredients = re.split(r'[,.\n:]', text)
        
        # Clean each ingredient
        cleaned_ingredients = []
        for ingredient in ingredients:
            ingredient = ingredient.strip()
            
            # Remove parenthetical content
            ingredient = re.sub(r'\([^)]*\)', '', ingredient).strip()
            
            # Skip very short or common non-ingredient words
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
        
        print(f"✅ Found {len(unique_ingredients)} unique ingredients")
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
        
        # Use 75% threshold for better detection
        if best_score >= 75:
            return True, best_match, best_score
        
        return False, None, 0
    
    def analyze_captured_image(self, image_path, product_name="", brand=""):
        """Run full analysis on captured image"""
        print(f"\n🔬 ANALYZING CAPTURED IMAGE")
        print("="*60)
        
        # Step 1: Extract text using OCR
        print("📝 Step 1: Extracting text from image...")
        extracted_text = self.extract_text_from_image(image_path)
        
        if not extracted_text:
            return {"error": "Could not extract text from image. Try better lighting or clearer image."}
        
        print(f"✅ Extracted {len(extracted_text)} characters of text")
        
        # Step 2: Extract ingredients
        print("\n🧪 Step 2: Identifying ingredients...")
        ingredients_list = self.parse_ingredients(extracted_text)
        
        if not ingredients_list:
            return {"error": "Could not identify ingredients from extracted text. Please ensure ingredient list is visible and clear."}
        
        print(f"✅ Found {len(ingredients_list)} ingredients:")
        for i, ingredient in enumerate(ingredients_list[:10], 1):  # Show first 10
            print(f"  {i}. {ingredient}")
        if len(ingredients_list) > 10:
            print(f"  ... and {len(ingredients_list) - 10} more")
        
        # Step 3: Detect microplastics
        print(f"\n🔬 Step 3: Analyzing for microplastics...")
        microplastic_ingredients = []
        safe_ingredients = []
        
        for ingredient in ingredients_list:
            is_microplastic, matched_ingredient, confidence = self.fuzzy_match_ingredient(ingredient)
            
            if is_microplastic:
                mp_data = self.microplastic_db[matched_ingredient]
                microplastic_ingredients.append({
                    'ingredient': ingredient,
                    'matched_as': matched_ingredient,
                    'confidence': confidence / 100,  # Convert to decimal
                    'risk_level': mp_data['risk_level'],
                    'health_concerns': mp_data['health_concerns'],
                    'category': mp_data['category']
                })
                print(f"⚠️  MICROPLASTIC DETECTED: {ingredient} → {matched_ingredient} ({confidence}%)")
            else:
                safe_ingredients.append(ingredient)
        
        # Step 4: Assess safety
        microplastic_count = len(microplastic_ingredients)
        total_count = len(ingredients_list)
        
        # Calculate safety score
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
        
        # Step 5: Prepare results
        results = {
            'product_name': product_name,
            'brand': brand,
            'image_path': image_path,
            'ingredients': ', '.join(ingredients_list),
            'microplastics_found': {
                'red': microplastic_ingredients,
                'orange': [],  # For consistency
                'green': [{'ingredient': ing} for ing in safe_ingredients]
            },
            'safety_rating': safety_rating,
            'risk_level': risk_level,
            'warnings': warnings,
            'extracted_text': extracted_text,
            'total_ingredients': len(ingredients_list),
            'safety_score': safety_score
        }
        
        # Step 6: Save to database
        self.save_to_database(results)
        print(f"💾 Results saved to database")
        
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
            
            print(f"✅ Analysis saved to database (Product ID: {product_id})")
            
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
    
    def display_analysis_results(self, results):
        """Display detailed analysis results"""
        if "error" in results:
            print(f"\n❌ ANALYSIS ERROR:")
            print(f"   {results['error']}")
            print(f"\n💡 TROUBLESHOOTING TIPS:")
            print(f"   - Ensure ingredient list is clearly visible")
            print(f"   - Use good lighting")
            print(f"   - Hold camera steady")
            print(f"   - Try capturing from different angles")
            return
        
        print("\n" + "="*60)
        print("🔬 MICROPLASTIC ANALYSIS RESULTS")
        print("="*60)
        
        print(f"🏷️  Product: {results['product_name'] or 'Unknown Product'}")
        print(f"🏢 Brand: {results['brand'] or 'Unknown Brand'}")
        print(f"📸 Image: {results['image_path']}")
        print(f"🧪 Total Ingredients: {results['total_ingredients']}")
        print(f"📊 Safety Score: {results['safety_score']}/100")
        
        print(f"\n⚖️  SAFETY ASSESSMENT:")
        print(f"   Rating: {results['safety_rating']}")
        print(f"   Risk Level: {results['risk_level']}")
        print(f"   Warning: {results['warnings']}")
        
        microplastics = results['microplastics_found']
        
        if microplastics['red']:
            print(f"\n🔴 HARMFUL Microplastics Found ({len(microplastics['red'])}):")
            for item in microplastics['red']:
                print(f"  • {item['ingredient']} (confidence: {item['confidence']:.1%})")
                print(f"    ↳ Matched as: {item['matched_as']}")
                print(f"    ↳ Category: {item['category']}")
                print(f"    ↳ Concerns: {', '.join(item['health_concerns'][:2])}")
        else:
            print(f"\n✅ NO MICROPLASTICS DETECTED!")
        
        if microplastics['green']:
            print(f"\n🟢 Safe Ingredients ({len(microplastics['green'])}):")
            safe_count = len(microplastics['green'])
            if safe_count <= 5:
                for item in microplastics['green']:
                    print(f"  • {item['ingredient']}")
            else:
                for item in microplastics['green'][:5]:
                    print(f"  • {item['ingredient']}")
                print(f"  • ... and {safe_count - 5} more safe ingredients")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if microplastics['red']:
            print("❌ NOT RECOMMENDED")
            print("  - Contains harmful microplastics")
            print("  - May contribute to environmental pollution")
            print("  - Consider finding alternatives")
        else:
            print("✅ SAFER CHOICE")
            print("  - No obvious harmful microplastics detected")
            print("  - Better option for you and environment")
        
        print("="*60)
    
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
    
    def run_capture_and_analyze(self):
        """Main function: Capture image and run full analysis"""
        print("🌟 CAMERA CAPTURE + MICROPLASTIC ANALYZER")
        print("="*60)
        print("🛠️  Enhanced for Windows camera compatibility!")
        
        while True:
            print("\n📋 OPTIONS:")
            print("1. 📸 Capture and analyze new product")
            print("2. 📚 View analysis history") 
            print("3. 🧪 Test camera")
            print("4. 🚪 Exit")
            
            try:
                choice = input("\nChoose option (1-4): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Thanks for using Microplastic Analyzer!")
                break
            
            if choice == '1':
                # Capture and analyze
                print("\n" + "="*40)
                captured_image = self.capture_image_from_camera()
                
                if captured_image is not None:
                    # Save image
                    image_path = self.save_captured_image(captured_image)
                    
                    # Get product details (optional)
                    print("\n📋 Product Information (optional):")
                    try:
                        product_name = input("🏷️  Product name: ").strip()
                        brand = input("🏢 Brand name: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        product_name = ""
                        brand = ""
                        print("\nUsing default product information...")
                    
                    # Run full analysis
                    print("\n🚀 Starting analysis...")
                    results = self.analyze_captured_image(image_path, product_name, brand)
                    
                    # Display results
                    self.display_analysis_results(results)
                    
                    # Save detailed results
                    if "error" not in results:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        results_path = f"captures/analysis_{timestamp}.json"
                        with open(results_path, 'w') as f:
                            json.dump(results, f, indent=2, default=str)
                        print(f"📄 Detailed results saved: {results_path}")
                else:
                    print("❌ No image captured. Please try again.")
            
            elif choice == '2':
                # Show history
                history = self.get_product_history()
                if not history:
                    print("📭 No analysis history found!")
                else:
                    print(f"\n📚 Analysis History ({len(history)} products):")
                    print("-" * 80)
                    for i, product in enumerate(history[:10], 1):  # Show last 10
                        product_name = product[1] or 'Unknown Product'
                        brand = product[2] or 'Unknown Brand'
                        safety_level = product[4]
                        safety_score = product[3]
                        microplastic_count = product[6]
                        scan_date = product[7][:16]  # Remove seconds
                        
                        # Color coding for safety level
                        if safety_level == 'GREEN':
                            status_icon = "✅"
                        elif safety_level == 'YELLOW':
                            status_icon = "⚠️ "
                        else:
                            status_icon = "❌"
                        
                        print(f"{i:2d}. {status_icon} {product_name} - {brand}")
                        print(f"     Score: {safety_score}/100 | Microplastics: {microplastic_count} | Date: {scan_date}")
                        print()
            
            elif choice == '3':
                # Test camera
                print("\n🧪 TESTING CAMERA WITH ENHANCED DETECTION")
                print("=" * 50)
                cap, camera_idx, backend = self.find_working_camera_advanced()
                if cap:
                    backend_name = {
                        cv2.CAP_DSHOW: "DirectShow",
                        cv2.CAP_MSMF: "Media Foundation",
                        cv2.CAP_ANY: "Auto"
                    }.get(backend, "Unknown")
                    
                    print(f"✅ Camera {camera_idx} working with {backend_name}")
                    print("📸 Taking test photo...")
                    
                    try:
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            test_path = f"captures/camera_test_{datetime.now().strftime('%H%M%S')}.jpg"
                            cv2.imwrite(test_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            print(f"✅ Test photo saved: {test_path}")
                            print("🔍 Check the saved image to verify camera is working")
                        else:
                            print("❌ Failed to capture test frame")
                    except Exception as e:
                        print(f"❌ Test capture error: {e}")
                    
                    cap.release()
                    self.safe_destroy_windows()
                else:
                    print("❌ No working camera found with any backend")
            
            elif choice == '4':
                print("👋 Thanks for using Microplastic Analyzer!")
                print("🌍 Keep making environmentally conscious choices!")
                break
            
            else:
                print("❌ Invalid choice! Please choose 1, 2, 3, or 4.")

def main():
    print("🌟 MICROPLASTIC DETECTION SYSTEM - CAMERA FIXED")
    print("="*60)
    
    # Check OpenCV installation
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
    except Exception as e:
        print(f"❌ OpenCV issue: {e}")
        print("💡 Try: pip install opencv-python")
        return
    
    print("🛠️  Enhanced camera handling for Windows compatibility!")
    
    try:
        analyzer = CameraCaptureAnalyzer()
        analyzer.run_capture_and_analyze()
    except KeyboardInterrupt:
        print("\n👋 Analyzer stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
