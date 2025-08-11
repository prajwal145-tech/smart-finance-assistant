"""
Smart Personal Finance Assistant - Financial Analyzer
===================================================

This module analyzes financial data and provides insights.
Think of it as your personal financial detective!

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialAnalyzer:
    """
    A class to analyze financial transactions and provide insights
    """
    
    def __init__(self, data_file=None, dataframe=None):
        """
        Initialize the analyzer with transaction data
        
        Args:
            data_file (str): Path to CSV file with transactions
            dataframe (pd.DataFrame): DataFrame with transaction data
        """
        if data_file:
            self.df = pd.read_csv(data_file)
        elif dataframe is not None:
            self.df = dataframe.copy()
        else:
            raise ValueError("Please provide either data_file or dataframe")
        
        # Clean and prepare the data
        self.prepare_data()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def prepare_data(self):
        """Clean and prepare the transaction data for analysis"""
        
        # Convert date column
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Create additional time-based columns
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['month_name'] = self.df['date'].dt.strftime('%B')
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        
        # Separate income and expenses
        self.df['is_income'] = self.df['amount'] > 0
        self.df['abs_amount'] = abs(self.df['amount'])
        
        # Get currency symbol
        self.currency = self.df['currency'].iloc[0] if 'currency' in self.df.columns else 'USD'
        self.currency_symbols = {'GBP': '£', 'INR': '₹', 'USD': '$'}
        self.symbol = self.currency_symbols.get(self.currency, '$')
        
        print(f"📊 Data loaded successfully!")
        print(f"📅 Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
        print(f"💰 Currency: {self.currency} ({self.symbol})")
        print(f"📈 Total transactions: {len(self.df)}")
    
    def get_overview_stats(self):
        """Get basic financial overview statistics"""
        
        total_income = self.df[self.df['amount'] > 0]['amount'].sum()
        total_expenses = abs(self.df[self.df['amount'] < 0]['amount'].sum())
        net_amount = total_income - total_expenses
        
        avg_monthly_income = total_income / self.df['month'].nunique()
        avg_monthly_expenses = total_expenses / self.df['month'].nunique()
        
        stats = {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_amount': net_amount,
            'avg_monthly_income': avg_monthly_income,
            'avg_monthly_expenses': avg_monthly_expenses,
            'savings_rate': (net_amount / total_income * 100) if total_income > 0 else 0,
            'num_transactions': len(self.df),
            'avg_transaction_size': self.df['abs_amount'].mean()
        }
        
        return stats
    
    def analyze_spending_by_category(self):
        """Analyze spending patterns by category"""
        
        # Get expenses only (negative amounts)
        expenses = self.df[self.df['amount'] < 0].copy()
        expenses['abs_amount'] = abs(expenses['amount'])
        
        # Group by category
        category_analysis = expenses.groupby('category').agg({
            'abs_amount': ['sum', 'mean', 'count'],
            'date': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        category_analysis.columns = ['total_spent', 'avg_per_transaction', 'num_transactions', 'first_transaction', 'last_transaction']
        
        # Calculate percentage of total spending
        total_expenses = category_analysis['total_spent'].sum()
        category_analysis['percentage_of_total'] = (category_analysis['total_spent'] / total_expenses * 100).round(1)
        
        # Sort by total spending
        category_analysis = category_analysis.sort_values('total_spent', ascending=False)
        
        return category_analysis
    
    def analyze_monthly_trends(self):
        """Analyze spending and income trends by month"""
        
        monthly_analysis = self.df.groupby(['year', 'month']).agg({
            'amount': lambda x: x[x > 0].sum(),  # Income
            'abs_amount': lambda x: x[x.index.isin(self.df[self.df['amount'] < 0].index)].sum()  # Expenses
        }).round(2)
        
        monthly_analysis.columns = ['income', 'expenses']
        monthly_analysis['net'] = monthly_analysis['income'] - monthly_analysis['expenses']
        monthly_analysis['savings_rate'] = (monthly_analysis['net'] / monthly_analysis['income'] * 100).round(1)
        
        # Add month names for better display
        monthly_analysis = monthly_analysis.reset_index()
        monthly_analysis['month_name'] = pd.to_datetime(monthly_analysis[['year', 'month']].assign(day=1)).dt.strftime('%B %Y')
        
        return monthly_analysis
    
    def find_unusual_transactions(self, threshold_multiplier=2.5):
        """Find potentially unusual or suspicious transactions"""
        
        unusual_transactions = []
        
        # Find transactions that are unusually large for each category
        for category in self.df['category'].unique():
            if category == 'Income':  # Skip income transactions
                continue
                
            category_data = self.df[self.df['category'] == category]['abs_amount']
            
            if len(category_data) > 1:  # Need at least 2 transactions to calculate stats
                mean_amount = category_data.mean()
                std_amount = category_data.std()
                threshold = mean_amount + (threshold_multiplier * std_amount)
                
                unusual_in_category = self.df[
                    (self.df['category'] == category) & 
                    (self.df['abs_amount'] > threshold)
                ].copy()
                
                if not unusual_in_category.empty:
                    unusual_in_category['reason'] = f'Unusually large {category.lower()} transaction'
                    unusual_in_category['typical_amount'] = mean_amount
                    unusual_transactions.append(unusual_in_category)
        
        if unusual_transactions:
            return pd.concat(unusual_transactions).sort_values('abs_amount', ascending=False)
        else:
            return pd.DataFrame()  # Return empty DataFrame if no unusual transactions
    
    def get_spending_insights(self):
        """Generate actionable spending insights"""
        
        insights = []
        
        # Get category analysis
        category_analysis = self.analyze_spending_by_category()
        monthly_analysis = self.analyze_monthly_trends()
        
        # Top spending category
        top_category = category_analysis.index[0]
        top_percentage = category_analysis.loc[top_category, 'percentage_of_total']
        insights.append(f"🔍 Your biggest expense is {top_category}, accounting for {top_percentage}% of your spending")
        
        # Savings rate insights
        avg_savings_rate = monthly_analysis['savings_rate'].mean()
        if avg_savings_rate > 20:
            insights.append(f"💰 Great job! Your average savings rate is {avg_savings_rate:.1f}% - that's excellent!")
        elif avg_savings_rate > 10:
            insights.append(f"💡 Your savings rate is {avg_savings_rate:.1f}% - consider aiming for 20% for optimal financial health")
        else:
            insights.append(f"⚠️ Your savings rate is {avg_savings_rate:.1f}% - try to reduce expenses or increase income")
        
        # Spending consistency
        monthly_expenses = monthly_analysis['expenses']
        expense_std = monthly_expenses.std()
        expense_mean = monthly_expenses.mean()
        cv = expense_std / expense_mean  # Coefficient of variation
        
        if cv < 0.1:
            insights.append("📊 Your spending is very consistent month-to-month - great for budgeting!")
        elif cv > 0.3:
            insights.append("📈 Your spending varies significantly by month - consider creating a more consistent budget")
        
        # Category-specific insights
        if 'Dining Out' in category_analysis.index:
            dining_percentage = category_analysis.loc['Dining Out', 'percentage_of_total']
            if dining_percentage > 15:
                potential_savings = category_analysis.loc['Dining Out', 'total_spent'] * 0.3
                insights.append(f"🍽️ You spend {dining_percentage}% on dining out - cooking more could save you {self.symbol}{potential_savings:.0f}")
        
        return insights
    
    def create_spending_summary_chart(self, save_path=None):
        """Create a visual summary of spending by category"""
        
        category_analysis = self.analyze_spending_by_category()
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('💰 Financial Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Spending by Category (Pie Chart)
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_analysis)))
        ax1.pie(category_analysis['total_spent'], labels=category_analysis.index, autopct='%1.1f%%', colors=colors)
        ax1.set_title('Spending by Category')
        
        # 2. Category Spending (Bar Chart)
        category_analysis.head(8)['total_spent'].plot(kind='barh', ax=ax2, color='skyblue')
        ax2.set_title('Top 8 Categories by Amount')
        ax2.set_xlabel(f'Amount ({self.symbol})')
        
        # 3. Monthly Trends
        monthly_analysis = self.analyze_monthly_trends()
        ax3.plot(monthly_analysis['month'], monthly_analysis['income'], marker='o', label='Income', linewidth=2)
        ax3.plot(monthly_analysis['month'], monthly_analysis['expenses'], marker='s', label='Expenses', linewidth=2)
        ax3.set_title('Monthly Income vs Expenses')
        ax3.set_xlabel('Month')
        ax3.set_ylabel(f'Amount ({self.symbol})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Savings Rate by Month
        ax4.bar(monthly_analysis['month'], monthly_analysis['savings_rate'], color='green', alpha=0.7)
        ax4.set_title('Monthly Savings Rate')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Savings Rate (%)')
        ax4.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Target: 20%')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved to: {save_path}")
        
        return fig
    
    def generate_monthly_report(self, year=2024, month=8):
        """Generate a detailed monthly financial report"""
        
        # Filter data for specific month
        month_data = self.df[(self.df['year'] == year) & (self.df['month'] == month)]
        
        if month_data.empty:
            return f"No data found for {year}-{month:02d}"
        
        # Calculate monthly statistics
        total_income = month_data[month_data['amount'] > 0]['amount'].sum()
        total_expenses = abs(month_data[month_data['amount'] < 0]['amount'].sum())
        net_amount = total_income - total_expenses
        
        # Category breakdown
        expenses_by_category = month_data[month_data['amount'] < 0].groupby('category')['abs_amount'].sum().sort_values(ascending=False)
        
        # Generate report
        month_name = month_data['month_name'].iloc[0]
        
        report = f"""
📊 MONTHLY FINANCIAL REPORT - {month_name} {year}
{'='*50}

💰 OVERVIEW:
   Total Income:     {self.symbol}{total_income:,.2f}
   Total Expenses:   {self.symbol}{total_expenses:,.2f}
   Net Amount:       {self.symbol}{net_amount:,.2f}
   Savings Rate:     {(net_amount/total_income*100):.1f}%

📈 TOP EXPENSE CATEGORIES:
"""
        
        for i, (category, amount) in enumerate(expenses_by_category.head(5).items(), 1):
            percentage = (amount / total_expenses) * 100
            report += f"   {i}. {category:<15} {self.symbol}{amount:>8,.2f} ({percentage:.1f}%)\n"
        
        # Add insights
        insights = self.get_spending_insights()
        report += f"\n💡 INSIGHTS:\n"
        for insight in insights[:3]:  # Top 3 insights
            report += f"   • {insight}\n"
        
        return report

# Example usage and testing
if __name__ == "__main__":
    print("📊 Smart Personal Finance Assistant - Financial Analyzer")
    print("=" * 60)
    
    # Test with sample data (you can replace with your CSV file)
    try:
        # Try to load generated data
        import os
        data_files = [
            "data/synthetic/uk_transactions_2024.csv",
            "data/synthetic/india_transactions_2024.csv"
        ]
        
        for data_file in data_files:
            if os.path.exists(data_file):
                print(f"\n🔍 Analyzing: {data_file}")
                analyzer = FinancialAnalyzer(data_file=data_file)
                
                # Get overview
                stats = analyzer.get_overview_stats()
                print(f"\n💰 OVERVIEW:")
                print(f"   Total Income: {analyzer.symbol}{stats['total_income']:,.2f}")
                print(f"   Total Expenses: {analyzer.symbol}{stats['total_expenses']:,.2f}")
                print(f"   Savings Rate: {stats['savings_rate']:.1f}%")
                
                # Get insights
                insights = analyzer.get_spending_insights()
                print(f"\n💡 INSIGHTS:")
                for insight in insights:
                    print(f"   • {insight}")
                
                # Generate monthly report
                print(analyzer.generate_monthly_report())
                
                break
        else:
            print("⚠️ No data files found. Run 'python src/data_generator.py' first!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to generate data first: python src/data_generator.py")