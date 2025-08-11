"""
Test our data generator - Run this first!
"""

# Import our data generator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import FinanceDataGenerator

def test_data_generation():
    """Test both UK and India data generation"""
    
    print("🚀 Testing Smart Finance Data Generator")
    print("=" * 50)
    
    # Test UK data generation
    print("\n🇬🇧 Testing UK Data Generation...")
    uk_gen = FinanceDataGenerator("UK")
    
    # Generate just 1 month of data for testing
    uk_transactions = uk_gen.generate_monthly_transactions(2024, 8, "medium_spender")
    print(f"✅ Generated {len(uk_transactions)} UK transactions for August 2024")
    
    # Show first few transactions
    print("\n📋 Sample UK Transactions:")
    for i, transaction in enumerate(uk_transactions[:5]):
        amount_str = f"£{transaction['amount']:,.2f}"
        print(f"{i+1}. {transaction['date']} | {amount_str:>10} | {transaction['category']:>12} | {transaction['merchant']}")
    
    print("\n" + "="*50)
    
    # Test India data generation
    print("\n🇮🇳 Testing India Data Generation...")
    india_gen = FinanceDataGenerator("India")
    
    india_transactions = india_gen.generate_monthly_transactions(2024, 8, "medium_spender")
    print(f"✅ Generated {len(india_transactions)} India transactions for August 2024")
    
    # Show first few transactions
    print("\n📋 Sample India Transactions:")
    for i, transaction in enumerate(india_transactions[:5]):
        amount_str = f"₹{transaction['amount']:,.2f}"
        print(f"{i+1}. {transaction['date']} | {amount_str:>12} | {transaction['category']:>12} | {transaction['merchant']}")
    
    print("\n🎉 Data Generator Test Complete!")
    print("\nNext Step: Generate full year data by running:")
    print("python src/data_generator.py")

if __name__ == "__main__":
    test_data_generation()