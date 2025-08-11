"""
Smart Personal Finance Assistant - AI Financial Advisor
=====================================================
This module integrates with AI APIs to provide personalized financial advice.
Uses OpenAI GPT for intelligent financial recommendations.

"""

import os
from typing import Dict, List, Optional
import json
import pandas as pd
from datetime import datetime

# For AI integration (optional - can be enabled later)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("💡 OpenAI not installed. Install with: pip install openai")

class AIFinancialAdvisor:
    """
    AI-powered financial advisor that provides personalized recommendations
    """
    
    def __init__(self, api_key: Optional[str] = None, use_ai: bool = False):
        """
        Initialize the AI Financial Advisor
        
        Args:
            api_key (str): OpenAI API key (optional)
            use_ai (bool): Whether to use real AI or smart rule-based system
        """
        self.use_ai = use_ai and OPENAI_AVAILABLE and api_key
        
        if self.use_ai:
            openai.api_key = api_key
            self.model = "gpt-3.5-turbo"
        
        # Smart rule-based advice system
        self.advice_rules = self._setup_advice_rules()
    
    def _setup_advice_rules(self) -> Dict:
        """Set up rule-based financial advice system"""
        return {
            "savings_rate": {
                "excellent": {"threshold": 25, "advice": "Outstanding! You're saving over 25% of your income. Consider investing in diversified funds or exploring higher-yield options."},
                "good": {"threshold": 15, "advice": "Good savings rate! Try to gradually increase to 20-25% for optimal financial security."},
                "fair": {"threshold": 10, "advice": "Your savings rate could improve. Look for areas to cut expenses or ways to increase income."},
                "poor": {"threshold": 0, "advice": "Critical: You need to build an emergency fund. Start with saving just £50-100 per month and gradually increase."}
            },
            "top_expense_categories": {
                "Groceries": "Consider meal planning and bulk buying to reduce grocery costs by 10-15%.",
                "Dining Out": "Dining out is a major expense. Try cooking more meals at home - you could save significantly!",
                "Shopping": "Review your shopping habits. Implement a 24-hour rule before non-essential purchases.",
                "Transport": "Look into public transport passes, carpooling, or cycling to reduce transport costs.",
                "Entertainment": "Set a monthly entertainment budget to keep leisure spending in check.",
                "Utilities": "Consider energy-efficient appliances and compare utility providers for better rates."
            },
            "unusual_spending": {
                "high_variance": "Your spending varies significantly month-to-month. Creating a more consistent budget could help you save more.",
                "large_transactions": "Be cautious of large, unusual transactions. Always verify these are necessary and planned expenses."
            },
            "income_expense_ratio": {
                "healthy": "Your income covers expenses comfortably. Focus on optimizing your investment strategy.",
                "tight": "Your expenses are close to your income. Look for ways to increase income or reduce non-essential costs.",
                "concerning": "You're spending more than you earn. This requires immediate attention - create an emergency budget plan."
            }
        }
    
    def analyze_financial_profile(self, stats: Dict, category_analysis: pd.DataFrame, 
                                monthly_analysis: pd.DataFrame, currency_symbol: str = "£") -> Dict:
        """
        Analyze complete financial profile and generate comprehensive advice
        
        Args:
            stats (Dict): Financial overview statistics
            category_analysis (pd.DataFrame): Spending by category analysis
            monthly_analysis (pd.DataFrame): Monthly trends analysis
            currency_symbol (str): Currency symbol for formatting
        
        Returns:
            Dict: Comprehensive financial advice and recommendations
        """
        
        profile = {
            "financial_health_score": self._calculate_health_score(stats),
            "savings_advice": self._get_savings_advice(stats),
            "spending_advice": self._get_spending_advice(category_analysis, stats['total_expenses']),
            "budgeting_recommendations": self._get_budgeting_advice(category_analysis, stats),
            "investment_suggestions": self._get_investment_advice(stats),
            "monthly_insights": self._analyze_monthly_patterns(monthly_analysis),
            "action_plan": self._create_action_plan(stats, category_analysis),
            "emergency_fund_status": self._assess_emergency_fund(stats),
            "currency": currency_symbol
        }
        
        return profile
    
    def _calculate_health_score(self, stats: Dict) -> Dict:
        """Calculate overall financial health score (0-100)"""
        
        score = 0
        factors = []
        
        # Savings rate (40% of score)
        savings_rate = stats.get('savings_rate', 0)
        if savings_rate >= 25:
            savings_score = 40
            factors.append("Excellent savings rate")
        elif savings_rate >= 15:
            savings_score = 30
            factors.append("Good savings rate")
        elif savings_rate >= 5:
            savings_score = 20
            factors.append("Fair savings rate")
        else:
            savings_score = 0
            factors.append("Poor savings rate - needs improvement")
        
        score += savings_score
        
        # Income stability (20% of score)
        if stats['total_income'] > 0:
            income_score = 20
            factors.append("Has income source")
        else:
            income_score = 0
            factors.append("No recorded income")
        
        score += income_score
        
        # Spending patterns (40% of score)
        expense_ratio = stats['total_expenses'] / stats['total_income'] if stats['total_income'] > 0 else 1.5
        if expense_ratio < 0.7:
            spending_score = 40
            factors.append("Conservative spending")
        elif expense_ratio < 0.9:
            spending_score = 25
            factors.append("Moderate spending")
        elif expense_ratio < 1.0:
            spending_score = 15
            factors.append("High spending")
        else:
            spending_score = 0
            factors.append("Spending exceeds income")
        
        score += spending_score
        
        # Determine health level
        if score >= 80:
            health_level = "Excellent"
            health_color = "🟢"
        elif score >= 60:
            health_level = "Good" 
            health_color = "🟡"
        elif score >= 40:
            health_level = "Fair"
            health_color = "🟠"
        else:
            health_level = "Needs Improvement"
            health_color = "🔴"
        
        return {
            "score": score,
            "level": health_level,
            "color": health_color,
            "factors": factors
        }
    
    def _get_savings_advice(self, stats: Dict) -> Dict:
        """Get personalized savings advice"""
        savings_rate = stats.get('savings_rate', 0)
        
        for level, data in self.advice_rules["savings_rate"].items():
            if savings_rate >= data["threshold"]:
                return {
                    "level": level,
                    "current_rate": savings_rate,
                    "advice": data["advice"],
                    "target_rate": 20 if savings_rate < 20 else savings_rate + 5,
                    "potential_monthly_savings": stats['avg_monthly_income'] * 0.05  # Suggest 5% improvement
                }
        
        return {
            "level": "poor",
            "current_rate": savings_rate,
            "advice": self.advice_rules["savings_rate"]["poor"]["advice"],
            "target_rate": 10,
            "potential_monthly_savings": stats['avg_monthly_income'] * 0.10
        }
    
    def _get_spending_advice(self, category_analysis: pd.DataFrame, total_expenses: float) -> List[Dict]:
        """Get category-specific spending advice"""
        advice_list = []
        
        # Focus on top 3 spending categories
        top_categories = category_analysis.head(3)
        
        for category, data in top_categories.iterrows():
            category_advice = {
                "category": category,
                "amount": data['total_spent'],
                "percentage": data['percentage_of_total'],
                "advice": self.advice_rules["top_expense_categories"].get(
                    category, 
                    f"Monitor your {category.lower()} spending and look for optimization opportunities."
                )
            }
            
            # Add specific recommendations based on percentage
            if data['percentage_of_total'] > 30:
                category_advice["priority"] = "High"
                category_advice["advice"] += f" This category represents {data['percentage_of_total']:.1f}% of your spending - significant savings potential here!"
            elif data['percentage_of_total'] > 20:
                category_advice["priority"] = "Medium" 
                category_advice["advice"] += " Consider setting a monthly budget limit for this category."
            else:
                category_advice["priority"] = "Low"
            
            advice_list.append(category_advice)
        
        return advice_list
    
    def _get_budgeting_advice(self, category_analysis: pd.DataFrame, stats: Dict) -> Dict:
        """Generate budgeting recommendations using 50/30/20 rule"""
        
        monthly_income = stats['avg_monthly_income']
        monthly_expenses = stats['avg_monthly_expenses']
        
        # 50/30/20 budgeting rule
        recommended_budget = {
            "needs": monthly_income * 0.50,  # Essential expenses
            "wants": monthly_income * 0.30,  # Non-essential
            "savings": monthly_income * 0.20  # Savings and debt repayment
        }
        
        # Categorize current expenses
        essential_categories = ['Groceries', 'Utilities', 'Transport', 'Healthcare', 'Insurance']
        current_needs = category_analysis[category_analysis.index.isin(essential_categories)]['total_spent'].sum() / 12
        current_wants = monthly_expenses - current_needs
        current_savings = monthly_income - monthly_expenses
        
        return {
            "recommended": recommended_budget,
            "current": {
                "needs": current_needs,
                "wants": current_wants, 
                "savings": current_savings
            },
            "adjustments_needed": {
                "needs": current_needs - recommended_budget["needs"],
                "wants": current_wants - recommended_budget["wants"],
                "savings": recommended_budget["savings"] - current_savings
            }
        }
    
    def _get_investment_advice(self, stats: Dict) -> Dict:
        """Provide investment recommendations based on financial situation"""
        
        net_worth = stats['net_amount']
        monthly_surplus = stats['avg_monthly_income'] - stats['avg_monthly_expenses']
        
        advice = {
            "emergency_fund_first": True,
            "recommended_amount": stats['avg_monthly_expenses'] * 6,
            "investment_readiness": "Not Ready",
            "suggestions": []
        }
        
        if monthly_surplus > 0:
            if net_worth >= stats['avg_monthly_expenses'] * 3:
                advice["emergency_fund_first"] = False
                advice["investment_readiness"] = "Ready"
                
                if monthly_surplus < 500:
                    advice["suggestions"] = [
                        "Start with low-cost index funds",
                        "Consider automatic monthly investments",
                        "Explore tax-advantaged accounts (ISA in UK, ELSS in India)"
                    ]
                elif monthly_surplus < 1500:
                    advice["suggestions"] = [
                        "Diversified portfolio with index funds",
                        "Consider real estate investment trusts (REITs)",
                        "Explore international diversification",
                        "Tax-efficient investment accounts"
                    ]
                else:
                    advice["suggestions"] = [
                        "Comprehensive portfolio diversification",
                        "Consider individual stocks alongside index funds", 
                        "Explore alternative investments",
                        "Consult with a financial advisor for advanced strategies"
                    ]
            else:
                advice["suggestions"] = [
                    "Focus on building emergency fund first",
                    "High-yield savings account for emergency funds",
                    "Start learning about investments for future"
                ]
        else:
            advice["suggestions"] = [
                "Focus on increasing income or reducing expenses first",
                "Create a debt repayment plan if applicable",
                "Build basic financial stability before investing"
            ]
        
        return advice
    
    def _analyze_monthly_patterns(self, monthly_analysis: pd.DataFrame) -> Dict:
        """Analyze monthly spending and income patterns"""
        
        if monthly_analysis.empty:
            return {"pattern": "No data", "insights": []}
        
        income_std = monthly_analysis['income'].std()
        expense_std = monthly_analysis['expenses'].std() 
        income_mean = monthly_analysis['income'].mean()
        expense_mean = monthly_analysis['expenses'].mean()
        
        insights = []
        
        # Income consistency
        if income_std / income_mean < 0.1:
            insights.append("✅ Very consistent income - great for budgeting!")
        elif income_std / income_mean > 0.3:
            insights.append("⚠️ Income varies significantly - consider building a larger emergency fund")
        
        # Expense consistency  
        if expense_std / expense_mean < 0.15:
            insights.append("✅ Consistent spending patterns - you have good budget control")
        elif expense_std / expense_mean > 0.4:
            insights.append("📊 Spending varies a lot month-to-month - try creating monthly budgets")
        
        # Seasonal patterns
        best_savings_month = monthly_analysis.loc[monthly_analysis['savings_rate'].idxmax(), 'month']
        worst_savings_month = monthly_analysis.loc[monthly_analysis['savings_rate'].idxmin(), 'month']
        
        insights.append(f"📈 Best savings month: Month {best_savings_month}")
        insights.append(f"📉 Watch spending in month {worst_savings_month}")
        
        return {
            "income_consistency": "high" if income_std/income_mean < 0.1 else "low",
            "expense_consistency": "high" if expense_std/expense_mean < 0.15 else "low", 
            "insights": insights,
            "avg_savings_rate": monthly_analysis['savings_rate'].mean()
        }
    
    def _create_action_plan(self, stats: Dict, category_analysis: pd.DataFrame) -> List[Dict]:
        """Create a personalized 30-60-90 day action plan"""
        
        action_plan = []
        savings_rate = stats.get('savings_rate', 0)
        
        # 30-day actions
        if savings_rate < 5:
            action_plan.append({
                "timeframe": "Next 30 Days",
                "priority": "High",
                "action": "Set up automatic savings",
                "description": "Start saving just £50-100 per month automatically",
                "expected_impact": "Begin building emergency fund"
            })
        
        top_category = category_analysis.index[0]
        action_plan.append({
            "timeframe": "Next 30 Days",
            "priority": "Medium", 
            "action": f"Track {top_category} spending",
            "description": f"Monitor and record all {top_category.lower()} expenses for one month",
            "expected_impact": "Identify specific areas for cost reduction"
        })
        
        # 60-day actions
        action_plan.append({
            "timeframe": "Next 60 Days",
            "priority": "Medium",
            "action": "Implement budget categories",
            "description": "Create monthly budgets for top 5 spending categories",
            "expected_impact": "Reduce overall expenses by 10-15%"
        })
        
        # 90-day actions
        if savings_rate >= 10:
            action_plan.append({
                "timeframe": "Next 90 Days", 
                "priority": "Low",
                "action": "Research investment options",
                "description": "Learn about index funds, ISAs, and investment accounts",
                "expected_impact": "Prepare for wealth building phase"
            })
        
        return action_plan
    
    def _assess_emergency_fund(self, stats: Dict) -> Dict:
        """Assess emergency fund status"""
        
        monthly_expenses = stats['avg_monthly_expenses']
        current_savings = stats['net_amount']
        recommended_fund = monthly_expenses * 6  # 6 months of expenses
        
        if current_savings >= recommended_fund:
            status = "Excellent"
            advice = "You have a solid emergency fund! Consider investing surplus funds."
        elif current_savings >= monthly_expenses * 3:
            status = "Good"
            advice = "You have a decent emergency fund. Aim for 6 months of expenses."
        elif current_savings >= monthly_expenses:
            status = "Fair" 
            advice = "You have some emergency savings. Try to build up to 3-6 months of expenses."
        else:
            status = "Poor"
            advice = "Priority: Build an emergency fund of at least 1 month's expenses."
        
        return {
            "status": status,
            "current_months": current_savings / monthly_expenses if monthly_expenses > 0 else 0,
            "recommended_months": 6,
            "shortfall": max(0, recommended_fund - current_savings),
            "advice": advice
        }
    
    def get_ai_powered_advice(self, financial_summary: str) -> str:
        """
        Get AI-powered financial advice (requires OpenAI API key)
        
        Args:
            financial_summary (str): Summary of user's financial situation
            
        Returns:
            str: AI-generated personalized advice
        """
        
        if not self.use_ai:
            return "AI advisor not enabled. Using rule-based recommendations instead."
        
        try:
            prompt = f"""
            You are a professional financial advisor. Based on the following financial summary, 
            provide personalized, actionable advice in a friendly but professional tone.
            Focus on practical steps the person can take to improve their financial situation.
            
            Financial Summary:
            {financial_summary}
            
            Please provide:
            1. Overall assessment
            2. Top 3 priority recommendations
            3. One specific action they can take this week
            
            Keep the response concise and encouraging.
            """
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"AI service temporarily unavailable: {str(e)}"

# Example usage and testing
if __name__ == "__main__":
    print("🤖 Smart Personal Finance Assistant - AI Financial Advisor")
    print("=" * 60)
    
    # Create advisor instance (without AI for testing)
    advisor = AIFinancialAdvisor(use_ai=False)
    
    # Example financial stats (you would get these from FinancialAnalyzer)
    example_stats = {
        'total_income': 45000,
        'total_expenses': 38000,
        'net_amount': 7000,
        'savings_rate': 15.6,
        'avg_monthly_income': 3750,
        'avg_monthly_expenses': 3167,
        'num_transactions': 234
    }
    
    # Example category analysis
    example_categories = pd.DataFrame({
        'total_spent': [12000, 8000, 6000, 5000, 4000],
        'percentage_of_total': [31.6, 21.1, 15.8, 13.2, 10.5]
    }, index=['Groceries', 'Transport', 'Utilities', 'Entertainment', 'Shopping'])
    
    # Example monthly analysis
    example_monthly = pd.DataFrame({
        'month': range(1, 13),
        'income': [3750] * 12,
        'expenses': [3200, 3100, 3300, 3150, 3200, 3100, 3250, 3180, 3220, 3300, 3400, 3500],
        'savings_rate': [14.7, 17.3, 12.0, 16.0, 14.7, 17.3, 13.3, 15.2, 14.1, 12.0, 9.3, 6.7]
    })
    
    # Get comprehensive analysis
    profile = advisor.analyze_financial_profile(
        example_stats, example_categories, example_monthly, "£"
    )
    
    # Display results
    print(f"\n💰 FINANCIAL HEALTH SCORE: {profile['financial_health_score']['score']}/100")
    print(f"Level: {profile['financial_health_score']['color']} {profile['financial_health_score']['level']}")
    
    print(f"\n📊 SAVINGS ADVICE:")
    print(f"Current Rate: {profile['savings_advice']['current_rate']:.1f}%")
    print(f"Target Rate: {profile['savings_advice']['target_rate']:.1f}%")
    print(f"Advice: {profile['savings_advice']['advice']}")
    
    print(f"\n🎯 ACTION PLAN:")
    for action in profile['action_plan']:
        print(f"• {action['timeframe']}: {action['action']} ({action['priority']} priority)")
    
    print(f"\n🚨 EMERGENCY FUND STATUS: {profile['emergency_fund_status']['status']}")
    print(f"Current: {profile['emergency_fund_status']['current_months']:.1f} months")
    print(f"Recommended: {profile['emergency_fund_status']['recommended_months']} months")