"""
Test our Financial Analyzer - Let's see the magic! ✨
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from financial_analyzer import FinancialAnalyzer

def test_financial_analysis():
    """Test our financial analysis capabilities"""
    
    print("🔍 Testing Smart Financial Analyzer")
    print("=" * 50)
    
    # Check if data files exist
    data_files = [
        "data/synthetic/uk_transactions_2024.csv",
        "data/synthetic/india_transactions_2024.csv"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"\n📊 Analyzing: {data_file}")
            print("-" * 40)
            
            # Create analyzer
            analyzer = FinancialAnalyzer(data_file=data_file)
            
            # Get basic overview
            stats = analyzer.get_overview_stats()
            print(f"\n💰 FINANCIAL OVERVIEW:")
            print(f"   Total Income:     {analyzer.symbol}{stats['total_income']:>10,.2f}")
            print(f"   Total Expenses:   {analyzer.symbol}{stats['total_expenses']:>10,.2f}")
            print(f"   Net Amount:       {analyzer.symbol}{stats['net_amount']:>10,.2f}")
            print(f"   Savings Rate:     {stats['savings_rate']:>13.1f}%")
            print(f"   Transactions:     {stats['num_transactions']:>13,}")
            
            # Category analysis
            print(f"\n📈 TOP SPENDING CATEGORIES:")
            category_analysis = analyzer.analyze_spending_by_category()
            for i, (category, data) in enumerate(category_analysis.head(5).iterrows(), 1):
                print(f"   {i}. {category:<15} {analyzer.symbol}{data['total_spent']:>8,.0f} ({data['percentage_of_total']:>4.1f}%)")
            
            # Get smart insights
            print(f"\n💡 SMART INSIGHTS:")
            insights = analyzer.get_spending_insights()
            for insight in insights:
                print(f"   • {insight}")
            
            # Find unusual transactions
            unusual = analyzer.find_unusual_transactions()
            if not unusual.empty:
                print(f"\n⚠️  UNUSUAL TRANSACTIONS DETECTED:")
                for _, transaction in unusual.head(3).iterrows():
                    print(f"   • {transaction['date'].strftime('%Y-%m-%d')} | {analyzer.symbol}{transaction['abs_amount']:.2f} | {transaction['category']} | {transaction['merchant']}")
            else:
                print(f"\n✅ No unusual transactions detected - all spending looks normal!")
            
            # Monthly report for current month
            print(f"\n📋 MONTHLY REPORT (August 2024):")
            report = analyzer.generate_monthly_report(2024, 8)
            print(report)
            
            break
    else:
        print("⚠️ No data files found!")
        print("💡 Run 'python src/data_generator.py' first to generate data")

if __name__ == "__main__":
    test_financial_analysis()