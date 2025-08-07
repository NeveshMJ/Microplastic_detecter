import json
import re
from difflib import SequenceMatcher

class IngredientDetector:
    def __init__(self, ingredients_db_path="data/microplastic_ingredients.json"):
        self.load_ingredients_database(ingredients_db_path)
    
    def load_ingredients_database(self, db_path):
        """Load microplastic ingredients database"""
        with open(db_path, 'r') as f:
            self.ingredients_db = json.load(f)
    
    def similarity(self, a, b):
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def find_microplastics(self, ingredients_list):
        """Detect microplastics in ingredients list"""
        detected_microplastics = {
            'red': [],
            'orange': [],
            'green': [],
            'unknown': []
        }
        
        for ingredient in ingredients_list:
            ingredient = ingredient.lower().strip()
            
            # Check against microplastic database
            for category in ['red_ingredients', 'orange_ingredients', 'green_ingredients']:
                for known_ingredient in self.ingredients_db[category]:
                    if self.is_ingredient_match(ingredient, known_ingredient):
                        category_name = category.split('_')[0]
                        detected_microplastics[category_name].append({
                            'ingredient': ingredient,
                            'matched_as': known_ingredient,
                            'confidence': self.similarity(ingredient, known_ingredient)
                        })
                        break
            
            # Check for common microplastic patterns
            if self.contains_microplastic_patterns(ingredient):
                detected_microplastics['red'].append({
                    'ingredient': ingredient,
                    'matched_as': 'pattern_detected',
                    'confidence': 0.8
                })
        
        return detected_microplastics
    
    def is_ingredient_match(self, ingredient, known_ingredient):
        """Check if ingredient matches known microplastic"""
        # Direct match
        if known_ingredient.lower() in ingredient.lower():
            return True
        
        # Similarity threshold
        if self.similarity(ingredient, known_ingredient) > 0.8:
            return True
        
        # Check for partial matches (e.g., "dimethicone" in "bis-aminopropyl dimethicone")
        if len(known_ingredient) > 8 and known_ingredient.lower() in ingredient.lower():
            return True
        
        return False
    
    def contains_microplastic_patterns(self, ingredient):
        """Check for common microplastic naming patterns"""
        patterns = [
            r'poly\w+',  # poly-anything
            r'\w*siloxane\w*',  # any siloxane
            r'\w*dimethicone\w*',  # any dimethicone variant
            r'\w*acrylate\w*',  # any acrylate
            r'\w*styrene\w*',  # any styrene
            r'peg-\d+',  # polyethylene glycol variants
        ]
        
        for pattern in patterns:
            if re.search(pattern, ingredient.lower()):
                return True
        return False
    
    def assess_safety(self, detected_microplastics):
        """Assess overall product safety"""
        red_count = len(detected_microplastics['red'])
        orange_count = len(detected_microplastics['orange'])
        
        if red_count > 0:
            risk_level = "HIGH RISK"
            safety_rating = "UNSAFE"
            warnings = f"Contains {red_count} harmful microplastic ingredient(s). May pose health risks and environmental damage."
        elif orange_count > 0:
            risk_level = "MODERATE RISK"
            safety_rating = "CAUTION"
            warnings = f"Contains {orange_count} questionable ingredient(s). Consider alternatives."
        else:
            risk_level = "LOW RISK"
            safety_rating = "SAFER CHOICE"
            warnings = "No known harmful microplastics detected."
        
        return {
            'risk_level': risk_level,
            'safety_rating': safety_rating,
            'warnings': warnings
        }