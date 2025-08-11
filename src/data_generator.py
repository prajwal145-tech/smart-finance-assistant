"""
Smart Personal Finance Assistant - Data Generator
==============================================

This module generates realistic financial transaction data for UK and India.
Perfect for testing and demo purposes!

Author: Prajwal Lawankar
Date: 2025
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import json
import os

class FinanceDataGenerator:
    """
    A class to generate realistic financial transactions for different countries
    """
    
    def __init__(self, country="UK"):
        """
        Initialize the data generator
        
        Args:
            country (str): "UK" or "India"
        """
        self.country = country
        self.fake = Faker()
        
        # Set up country-specific data
        if country == "UK":
            self.currency = "GBP"
            self.currency_symbol = "£"
            self.salary_range = (25000, 80000)  # Annual salary in GBP
            self.setup_uk_data()
        else:  # India
            self.currency = "INR"
            self.currency_symbol = "₹"
            self.salary_range = (300000, 1500000)  # Annual salary in INR
            self.setup_india_data()
    
    def setup_uk_data(self):
        """Set up UK-specific spending categories and merchants"""
        self.categories = {
            "Groceries": {
                "merchants": ["Tesco", "ASDA", "Sainsbury's", "Morrisons", "Iceland", "Aldi", "Lidl"],
                "amount_range": (15, 120),
                "frequency": 0.25  # 25% of all transactions
            },
            "Transport": {
                "merchants": ["TfL", "National Rail", "Shell", "BP", "Tesco Petrol", "First Bus"],
                "amount_range": (3, 80),
                "frequency": 0.15
            },
            "Dining Out": {
                "merchants": ["McDonald's", "Nando's", "Pizza Express", "Wagamama", "Local Pub", "Costa Coffee", "Starbucks"],
                "amount_range": (8, 45),
                "frequency": 0.12
            },
            "Utilities": {
                "merchants": ["British Gas", "EDF Energy", "Thames Water", "Sky", "BT", "Virgin Media"],
                "amount_range": (25, 150),
                "frequency": 0.08
            },
            "Entertainment": {
                "merchants": ["Netflix", "Spotify", "Amazon Prime", "Cinema", "Gym Membership", "Football Tickets"],
                "amount_range": (10, 60),
                "frequency": 0.10
            },
            "Shopping": {
                "merchants": ["Amazon UK", "John Lewis", "M&S", "H&M", "Zara", "Next", "Boots"],
                "amount_range": (20, 200),
                "frequency": 0.15
            },
            "Healthcare": {
                "merchants": ["NHS Prescription", "Private GP", "Dentist", "Opticians", "Pharmacy"],
                "amount_range": (10, 80),
                "frequency": 0.05
            },
            "Salary": {
                "merchants": ["Company Ltd", "Employer Inc", "Workplace PLC"],
                "amount_range": (2000, 6000),  # Monthly
                "frequency": 0.02
            },
            "Other": {
                "merchants": ["ATM Withdrawal", "Bank Transfer", "Online Purchase", "Miscellaneous"],
                "amount_range": (5, 100),
                "frequency": 0.08
            }
        }
    
    def setup_india_data(self):
        """Set up India-specific spending categories and merchants"""
        self.categories = {
            "Groceries": {
                "merchants": ["Big Bazaar", "Reliance Fresh", "Spencer's", "More Supermarket", "Local Kirana", "DMart"],
                "amount_range": (200, 2500),
                "frequency": 0.25
            },
            "Transport": {
                "merchants": ["BMTC", "Delhi Metro", "Ola", "Uber", "Indian Oil", "HP Petrol", "Auto Rickshaw"],
                "amount_range": (20, 800),
                "frequency": 0.15
            },
            "Dining Out": {
                "merchants": ["McDonald's India", "KFC", "Domino's", "Swiggy", "Zomato", "Local Restaurant", "CCD"],
                "amount_range": (100, 800),
                "frequency": 0.12
            },
            "Utilities": {
                "merchants": ["BESCOM", "Reliance Jio", "Airtel", "BWSSB", "Gas Agency", "DTH Recharge"],
                "amount_range": (300, 2000),
                "frequency": 0.08
            },
            "Entertainment": {
                "merchants": ["Netflix India", "Hotstar", "Amazon Prime", "PVR Cinemas", "Gym", "BookMyShow"],
                "amount_range": (150, 1000),
                "frequency": 0.10
            },
            "Shopping": {
                "merchants": ["Amazon India", "Flipkart", "Myntra", "Local Mall", "Brand Factory", "Medical Store"],
                "amount_range": (300, 5000),
                "frequency": 0.15
            },
            "Healthcare": {
                "merchants": ["Apollo Pharmacy", "Local Clinic", "Hospital", "Medical Tests", "Doctor Visit"],
                "amount_range": (200, 1500),
                "frequency": 0.05
            },
            "Salary": {
                "merchants": ["Tech Company", "MNC Corp", "Local Business"],
                "amount_range": (25000, 120000),  # Monthly
                "frequency": 0.02
            },
            "Other": {
                "merchants": ["ATM", "UPI Transfer", "Online Payment", "Miscellaneous"],
                "amount_range": (50, 2000),
                "frequency": 0.08
            }
        }
    
    def generate_monthly_transactions(self, year=2024, month=1, profile="medium_spender"):
        """
        Generate transactions for a specific month
        
        Args:
            year (int): Year for transactions
            month (int): Month (1-12)
            profile (str): "student", "medium_spender", "family", "high_earner"
        
        Returns:
            list: List of transaction dictionaries
        """
        transactions = []
        
        # Adjust spending based on profile
        multipliers = {
            "student": 0.3,
            "medium_spender": 1.0,
            "family": 1.8,
            "high_earner": 2.5
        }
        multiplier = multipliers.get(profile, 1.0)
        
        # Calculate number of days in month
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        
        current_month = datetime(year, month, 1)
        days_in_month = (next_month - current_month).days
        
        # Add salary (usually on 1st or last day)
        salary_day = random.choice([1, days_in_month])
        salary_amount = random.uniform(*self.categories["Salary"]["amount_range"]) * multiplier
        
        transactions.append({
            "date": datetime(year, month, salary_day).strftime("%Y-%m-%d"),
            "amount": round(salary_amount, 2),
            "category": "Income",
            "merchant": random.choice(self.categories["Salary"]["merchants"]),
            "description": "Monthly Salary",
            "type": "credit",
            "currency": self.currency
        })
        
        # Generate daily transactions
        for day in range(1, days_in_month + 1):
            # Number of transactions per day (0-4, weighted towards 1-2)
            num_transactions = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.4, 0.3, 0.15, 0.05])
            
            for _ in range(num_transactions):
                # Choose category based on frequency
                category = np.random.choice(
                    list(self.categories.keys()),
                    p=[self.categories[cat]["frequency"] for cat in self.categories.keys()]
                )
                
                if category == "Salary":  # Skip extra salary transactions
                    continue
                
                # Generate transaction details
                merchant = random.choice(self.categories[category]["merchants"])
                amount_min, amount_max = self.categories[category]["amount_range"]
                amount = random.uniform(amount_min, amount_max) * multiplier
                
                # Add some randomness to amounts
                amount = amount * random.uniform(0.8, 1.2)
                
                transactions.append({
                    "date": datetime(year, month, day).strftime("%Y-%m-%d"),
                    "amount": -round(amount, 2),  # Negative for expenses
                    "category": category,
                    "merchant": merchant,
                    "description": f"{category} at {merchant}",
                    "type": "debit",
                    "currency": self.currency
                })
        
        return transactions
    
    def generate_year_data(self, year=2024, profile="medium_spender"):
        """
        Generate a full year of transaction data
        
        Args:
            year (int): Year to generate data for
            profile (str): Spending profile type
        
        Returns:
            pd.DataFrame: Complete year of transactions
        """
        all_transactions = []
        
        print(f"Generating {year} transaction data for {self.country} ({profile} profile)...")
        
        for month in range(1, 13):
            month_transactions = self.generate_monthly_transactions(year, month, profile)
            all_transactions.extend(month_transactions)
            print(f"✓ Generated {len(month_transactions)} transactions for {year}-{month:02d}")
        
        df = pd.DataFrame(all_transactions)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add running balance
        df['balance'] = df['amount'].cumsum()
        
        print(f"\n🎉 Successfully generated {len(df)} transactions!")
        print(f"💰 Total Income: {self.currency_symbol}{df[df['amount'] > 0]['amount'].sum():,.2f}")
        print(f"💸 Total Expenses: {self.currency_symbol}{abs(df[df['amount'] < 0]['amount'].sum()):,.2f}")
        print(f"💵 Net Amount: {self.currency_symbol}{df['amount'].sum():,.2f}")
        
        return df
    
    def save_data(self, df, filename=None):
        """Save generated data to CSV file"""
        if filename is None:
            filename = f"synthetic_transactions_{self.country.lower()}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Ensure data directory exists
        os.makedirs("data/synthetic", exist_ok=True)
        
        filepath = os.path.join("data", "synthetic", filename)
        df.to_csv(filepath, index=False)
        
        print(f"💾 Data saved to: {filepath}")
        return filepath

# Example usage and testing
if __name__ == "__main__":
    print("🏦 Smart Personal Finance Assistant - Data Generator")
    print("=" * 50)
    
    # Generate UK data
    uk_generator = FinanceDataGenerator("UK")
    uk_data = uk_generator.generate_year_data(2024, "medium_spender")
    uk_generator.save_data(uk_data, "uk_transactions_2024.csv")
    
    print("\n" + "=" * 50)
    
    # Generate India data
    india_generator = FinanceDataGenerator("India")
    india_data = india_generator.generate_year_data(2024, "medium_spender")
    india_generator.save_data(india_data, "india_transactions_2024.csv")
    
    print("\n✅ Data generation complete! Check the data/synthetic/ folder.")