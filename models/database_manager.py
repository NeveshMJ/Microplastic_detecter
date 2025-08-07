import sqlite3
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="data/scanned_products.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scanned_products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT,
                brand TEXT,
                ingredients TEXT,
                microplastics_found TEXT,
                safety_rating TEXT,
                scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                risk_level TEXT,
                warnings TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_product(self, product_data):
        """Save scanned product to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO scanned_products 
            (product_name, brand, ingredients, microplastics_found, 
             safety_rating, risk_level, warnings)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            product_data['product_name'],
            product_data['brand'],
            product_data['ingredients'],
            json.dumps(product_data['microplastics_found']),
            product_data['safety_rating'],
            product_data['risk_level'],
            product_data['warnings']
        ))
        
        conn.commit()
        conn.close()
    
    def get_product_history(self):
        """Get all scanned products"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM scanned_products ORDER BY scan_date DESC')
        products = cursor.fetchall()
        
        conn.close()
        return products