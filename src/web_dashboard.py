"""
Smart Personal Finance Assistant - Web Dashboard
===============================================

Beautiful, interactive web interface for financial analysis!
Built with Streamlit - turns Python into a web app instantly!

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import io

# Add src to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from financial_analyzer import FinancialAnalyzer

# Configure Streamlit page
st.set_page_config(
    page_title="💰 Smart Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load sample data from our generated files"""
    data_files = {
        "UK Sample Data (2024)": "data/synthetic/uk_transactions_2024.csv",
        "India Sample Data (2024)": "data/synthetic/india_transactions_2024.csv"
    }
    
    available_files = {}
    for name, path in data_files.items():
        if os.path.exists(path):
            available_files[name] = path
    
    return available_files

def create_overview_metrics(analyzer):
    """Create overview metrics display"""
    stats = analyzer.get_overview_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Income",
            value=f"{analyzer.symbol}{stats['total_income']:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="💸 Total Expenses", 
            value=f"{analyzer.symbol}{stats['total_expenses']:,.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="💵 Net Amount",
            value=f"{analyzer.symbol}{stats['net_amount']:,.2f}",
            delta=f"{stats['savings_rate']:.1f}% savings rate",
            delta_color="normal" if stats['savings_rate'] > 10 else "off"
        )
    
    with col4:
        st.metric(
            label="📈 Transactions",
            value=f"{stats['num_transactions']:,}",
            delta=f"Avg: {analyzer.symbol}{stats['avg_transaction_size']:,.0f}"
        )

def create_spending_charts(analyzer):
    """Create interactive spending visualization charts"""
    
    # Get category analysis
    category_analysis = analyzer.analyze_spending_by_category()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 Spending by Category")
        
        # Pie chart
        fig_pie = px.pie(
            values=category_analysis['total_spent'],
            names=category_analysis.index,
            title="Where Your Money Goes",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📊 Top Categories")
        
        # Bar chart
        top_categories = category_analysis.head(8)
        fig_bar = px.bar(
            x=top_categories['total_spent'],
            y=top_categories.index,
            orientation='h',
            title="Top 8 Spending Categories",
            labels={'x': f'Amount ({analyzer.symbol})', 'y': 'Category'},
            color=top_categories['total_spent'],
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

def create_monthly_trends(analyzer):
    """Create monthly trend analysis"""
    st.subheader("📈 Monthly Trends")
    
    monthly_analysis = analyzer.analyze_monthly_trends()
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly Income vs Expenses', 'Monthly Savings Rate'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Income vs Expenses
    fig.add_trace(
        go.Scatter(
            x=monthly_analysis['month'], 
            y=monthly_analysis['income'],
            name='Income',
            line=dict(color='green', width=3),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_analysis['month'], 
            y=monthly_analysis['expenses'],
            name='Expenses', 
            line=dict(color='red', width=3),
            mode='lines+markers'
        ),
        row=1, col=1
    )
    
    # Savings Rate
    fig.add_trace(
        go.Bar(
            x=monthly_analysis['month'],
            y=monthly_analysis['savings_rate'],
            name='Savings Rate (%)',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    # Add target line
    fig.add_hline(
        y=20, line_dash="dash", line_color="red",
        annotation_text="Target: 20%", row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text=f"Amount ({analyzer.symbol})", row=1, col=1)
    fig.update_yaxes(title_text="Savings Rate (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def display_insights(analyzer):
    """Display AI-like insights"""
    st.subheader("🤖 Smart Insights")
    
    insights = analyzer.get_spending_insights()
    
    for insight in insights:
        if "Great job" in insight or "excellent" in insight.lower():
            st.markdown(f'<div class="success-box">✅ {insight}</div>', unsafe_allow_html=True)
        elif "try to reduce" in insight.lower() or "consider" in insight.lower():
            st.markdown(f'<div class="warning-box">⚠️ {insight}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight-box">💡 {insight}</div>', unsafe_allow_html=True)

def display_unusual_transactions(analyzer):
    """Display unusual transactions if any"""
    unusual = analyzer.find_unusual_transactions()
    
    if not unusual.empty:
        st.subheader("🚨 Unusual Transactions Detected")
        
        # Display top unusual transactions
        unusual_display = unusual[['date', 'amount', 'category', 'merchant', 'reason']].head(5)
        unusual_display['amount'] = unusual_display['amount'].apply(lambda x: f"{analyzer.symbol}{abs(x):.2f}")
        
        st.dataframe(unusual_display, use_container_width=True)
    else:
        st.success("✅ All transactions look normal - no unusual patterns detected!")

def create_monthly_report(analyzer):
    """Create detailed monthly report"""
    st.subheader("📋 Monthly Report")
    
    # Month selector
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Select Year", [2024], index=0)
    with col2:
        month = st.selectbox("Select Month", list(range(1, 13)), index=7, format_func=lambda x: datetime(2024, x, 1).strftime('%B'))
    
    if st.button("Generate Report"):
        report = analyzer.generate_monthly_report(year, month)
        st.text(report)

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">💰 Smart Personal Finance Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📊 Data Source")
    
    # Data loading options
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["📁 Upload CSV File", "🎯 Use Sample Data"]
    )
    
    analyzer = None
    
    if data_option == "📁 Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your transaction CSV",
            type=['csv'],
            help="Upload a CSV file with columns: date, amount, category, merchant, description"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                analyzer = FinancialAnalyzer(dataframe=df)
                st.sidebar.success("✅ File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {str(e)}")
    
    else:  # Use sample data
        available_files = load_sample_data()
        
        if available_files:
            selected_file = st.sidebar.selectbox(
                "Choose sample dataset:",
                list(available_files.keys())
            )
            
            if st.sidebar.button("Load Sample Data"):
                try:
                    analyzer = FinancialAnalyzer(data_file=available_files[selected_file])
                    st.sidebar.success("✅ Sample data loaded!")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading sample data: {str(e)}")
        else:
            st.sidebar.warning("⚠️ No sample data found. Run the data generator first!")
            st.sidebar.code("python src/data_generator.py")
    
    # Main content
    if analyzer is not None:
        # Overview metrics
        st.header("📊 Financial Overview")
        create_overview_metrics(analyzer)
        
        st.markdown("---")
        
        # Spending analysis
        st.header("💸 Spending Analysis")
        create_spending_charts(analyzer)
        
        st.markdown("---")
        
        # Monthly trends
        create_monthly_trends(analyzer)
        
        st.markdown("---")
        
        # Insights
        display_insights(analyzer)
        
        st.markdown("---")
        
        # Unusual transactions
        display_unusual_transactions(analyzer)
        
        st.markdown("---")
        
        # Monthly report
        create_monthly_report(analyzer)
        
    else:
        # Welcome screen
        st.info("👆 Please select a data source from the sidebar to get started!")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://via.placeholder.com/400x300/1f77b4/white?text=Smart+Finance+Assistant", 
                    caption="Your Personal Finance AI Assistant")
        
        st.markdown("""
        ### 🌟 Features:
        - 📊 **Interactive Charts** - Beautiful visualizations of your spending
        - 🤖 **Smart Insights** - AI-powered financial recommendations  
        - 📈 **Trend Analysis** - Track your financial progress over time
        - 🚨 **Anomaly Detection** - Spot unusual transactions automatically
        - 📱 **Mobile Friendly** - Works on all devices
        - 🔒 **Privacy First** - Your data stays on your device
        
        ### 🚀 Get Started:
        1. Upload your bank CSV file, or
        2. Try our sample data to see the features
        3. Explore your financial insights!
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Python, Streamlit & AI | "
        "💡 **Pro Tip**: Export your bank statements as CSV to analyze your real data!"
    )

if __name__ == "__main__":
    main()