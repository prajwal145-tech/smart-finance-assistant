"""
Smart Personal Finance Assistant - Enhanced Main Application
==========================================================

This is the main application file that brings everything together:
- Beautiful web interface
- AI-powered insights
- ML predictions
- Financial analysis
- Multi-country support

Run with: streamlit run app.py

Author: Your Name
Date: 2024
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
import json

# Add src to path
sys.path.append('src')

# Import our modules
try:
    from financial_analyzer import FinancialAnalyzer
    from ai_financial_advisor import AIFinancialAdvisor
    from ml_financial_predictor import MLFinancialPredictor
    from data_generator import FinanceDataGenerator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Make sure all files are in the 'src' folder!")
    st.stop()

# Configure Streamlit
st.set_page_config(
    page_title="💰 Smart Finance Assistant Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .success-insight {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    
    .warning-insight {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    
    .ai-response {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
        transform: translateY(-10px);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'ai_advisor' not in st.session_state:
        st.session_state.ai_advisor = AIFinancialAdvisor()
    if 'ml_predictor' not in st.session_state:
        st.session_state.ml_predictor = None
    if 'current_data_source' not in st.session_state:
        st.session_state.current_data_source = None

def load_sample_data():
    """Load available sample data files"""
    data_files = {
        "🇬🇧 UK Sample Data (2024)": "data/synthetic/uk_transactions_2024.csv",
        "🇮🇳 India Sample Data (2024)": "data/synthetic/india_transactions_2024.csv"
    }
    
    available_files = {}
    for name, path in data_files.items():
        if os.path.exists(path):
            available_files[name] = path
    
    return available_files

def create_enhanced_metrics(analyzer):
    """Create enhanced overview metrics with animations"""
    stats = analyzer.get_overview_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: #1f77b4; margin-bottom: 0.5rem;">💰 Total Income</h3>
        <h2 style="color: #2e8b57; margin: 0;">{}{:,.2f}</h2>
        <p style="color: #666; margin-top: 0.5rem;">Monthly Avg: {}{:,.0f}</p>
        </div>
        """.format(analyzer.symbol, stats['total_income'], 
                  analyzer.symbol, stats['avg_monthly_income']), 
        unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: #1f77b4; margin-bottom: 0.5rem;">💸 Total Expenses</h3>
        <h2 style="color: #dc143c; margin: 0;">{}{:,.2f}</h2>
        <p style="color: #666; margin-top: 0.5rem;">Monthly Avg: {}{:,.0f}</p>
        </div>
        """.format(analyzer.symbol, stats['total_expenses'], 
                  analyzer.symbol, stats['avg_monthly_expenses']), 
        unsafe_allow_html=True)
    
    with col3:
        savings_color = "#2e8b57" if stats['net_amount'] > 0 else "#dc143c"
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: #1f77b4; margin-bottom: 0.5rem;">💵 Net Savings</h3>
        <h2 style="color: {}; margin: 0;">{}{:,.2f}</h2>
        <p style="color: #666; margin-top: 0.5rem;">Rate: {:.1f}%</p>
        </div>
        """.format(savings_color, analyzer.symbol, stats['net_amount'], 
                  stats['savings_rate']), 
        unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: #1f77b4; margin-bottom: 0.5rem;">📈 Transactions</h3>
        <h2 style="color: #4169e1; margin: 0;">{:,}</h2>
        <p style="color: #666; margin-top: 0.5rem;">Avg: {}{:,.0f}</p>
        </div>
        """.format(stats['num_transactions'], analyzer.symbol, 
                  stats['avg_transaction_size']), 
        unsafe_allow_html=True)

def create_advanced_charts(analyzer):
    """Create advanced interactive charts"""
    
    st.subheader("📊 Advanced Financial Analytics")
    
    # Tab layout for different chart types
    tab1, tab2, tab3, tab4 = st.tabs(["🥧 Category Analysis", "📈 Trends", "🔍 Deep Dive", "🤖 ML Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced pie chart
            category_analysis = analyzer.analyze_spending_by_category()
            fig_pie = px.pie(
                values=category_analysis['total_spent'],
                names=category_analysis.index,
                title="💰 Spending Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Amount: %{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
            )
            fig_pie.update_layout(
                showlegend=True,
                height=500,
                font=dict(size=12)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Create sunburst data properly
            try:
                # Create a DataFrame for the sunburst chart
                sunburst_data = pd.DataFrame({
                    'category': category_analysis.index,
                    'amount': category_analysis['total_spent'].values
                })
                
                # Create sunburst chart with proper data structure
                fig_sunburst = px.sunburst(
                    sunburst_data,
                    path=['category'],
                    values='amount',
                    title="🌅 Category Hierarchy",
                    color='amount',
                    color_continuous_scale='Viridis'
                )
                fig_sunburst.update_layout(height=500)
                st.plotly_chart(fig_sunburst, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating sunburst chart: {e}")
                # Fallback: show a simple bar chart instead
                fig_backup = px.bar(
                    x=category_analysis.index,
                    y=category_analysis['total_spent'],
                    title="🌅 Category Breakdown (Bar Chart)",
                    color=category_analysis['total_spent'],
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_backup, use_container_width=True)
    
    with tab2:
        try:
            # Enhanced monthly trends
            monthly_analysis = analyzer.analyze_monthly_trends()
            
            fig_trends = make_subplots(
                rows=2, cols=1,
                subplot_titles=('📊 Monthly Income vs Expenses', '💰 Savings Rate Trend'),
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
                vertical_spacing=0.12
            )
            
            # Income and expenses
            fig_trends.add_trace(
                go.Scatter(
                    x=monthly_analysis['month'], 
                    y=monthly_analysis['income'],
                    name='💰 Income',
                    line=dict(color='#2e8b57', width=4),
                    mode='lines+markers',
                    marker=dict(size=8),
                    hovertemplate='Income: %{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig_trends.add_trace(
                go.Scatter(
                    x=monthly_analysis['month'], 
                    y=monthly_analysis['expenses'],
                    name='💸 Expenses',
                    line=dict(color='#dc143c', width=4),
                    mode='lines+markers',
                    marker=dict(size=8),
                    hovertemplate='Expenses: %{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Savings rate with color gradient
            colors = ['red' if x < 0 else 'orange' if x < 10 else 'green' for x in monthly_analysis['savings_rate']]
            
            fig_trends.add_trace(
                go.Bar(
                    x=monthly_analysis['month'],
                    y=monthly_analysis['savings_rate'],
                    name='💵 Savings Rate (%)',
                    marker_color=colors,
                    hovertemplate='Savings Rate: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Target line
            fig_trends.add_hline(
                y=20, line_dash="dash", line_color="blue",
                annotation_text="🎯 Target: 20%", row=2, col=1
            )
            
            fig_trends.update_layout(height=700, showlegend=True)
            st.plotly_chart(fig_trends, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating trends chart: {e}")
    
    with tab3:
        try:
            # Deep dive analytics
            col1, col2 = st.columns(2)
            
            with col1:
                # Day of week spending
                expenses_df = analyzer.df[analyzer.df['amount'] < 0].copy()
                if not expenses_df.empty:
                    day_spending = expenses_df.groupby('day_of_week')['abs_amount'].sum()
                    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    # Map day numbers to names
                    day_spending_named = pd.Series(index=day_names, dtype=float)
                    for i, day_name in enumerate(day_names):
                        day_spending_named[day_name] = day_spending.get(i, 0)
                    
                    fig_days = px.bar(
                        x=day_names,
                        y=day_spending_named.values,
                        title="📅 Spending by Day of Week",
                        color=day_spending_named.values,
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_days, use_container_width=True)
                else:
                    st.info("No expense data available for day-of-week analysis")
            
            with col2:
                # Transaction size distribution
                expenses_df = analyzer.df[analyzer.df['amount'] < 0].copy()
                if not expenses_df.empty:
                    expense_amounts = expenses_df['abs_amount']
                    
                    fig_dist = px.histogram(
                        x=expense_amounts,
                        nbins=20,
                        title="💰 Transaction Size Distribution",
                        labels={'x': f'Amount ({analyzer.symbol})', 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.info("No expense data available for transaction size analysis")
        except Exception as e:
            st.error(f"Error creating deep dive charts: {e}")
    
    with tab4:
        # ML Insights
        try:
            if st.session_state.ml_predictor is not None:
                st.subheader("🤖 Machine Learning Predictions")
                
                # Get ML insights
                ml_insights = st.session_state.ml_predictor.get_spending_insights_ml()
                
                for insight in ml_insights:
                    if insight['impact'] == 'high':
                        st.markdown(f"""
                        <div class="warning-insight">
                        <h4>🚨 {insight['title']}</h4>
                        <p>{insight['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="insight-box">
                        <h4>💡 {insight['title']}</h4>
                        <p>{insight['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Spending forecast
                forecast = st.session_state.ml_predictor.create_spending_forecast_chart()
                if forecast is not None and not forecast.empty:
                    fig_forecast = px.bar(
                        forecast,
                        x='month',
                        y='predicted_expenses',
                        title="🔮 6-Month Spending Forecast",
                        color='predicted_expenses',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)
            else:
                st.info("🤖 Load data to enable ML predictions!")
        except Exception as e:
            st.error(f"Error in ML insights: {e}")

def display_ai_insights(analyzer):
    """Display AI-powered financial insights"""
    
    st.subheader("🤖 AI Financial Advisor")
    
    try:
        # Get comprehensive analysis
        stats = analyzer.get_overview_stats()
        category_analysis = analyzer.analyze_spending_by_category()
        monthly_analysis = analyzer.analyze_monthly_trends()
        
        # Generate AI profile
        ai_profile = st.session_state.ai_advisor.analyze_financial_profile(
            stats, category_analysis, monthly_analysis, analyzer.symbol
        )
        
        # Financial Health Score
        health_score = ai_profile['financial_health_score']
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 1rem; margin: 1rem 0; color: white;">
            <h2>{health_score['color']} Financial Health Score</h2>
            <h1 style="font-size: 4rem; margin: 1rem 0;">{health_score['score']}/100</h1>
            <h3>{health_score['level']}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💡 Savings Advice")
            savings_advice = ai_profile['savings_advice']
            
            advice_color = "success-insight" if savings_advice['current_rate'] > 15 else "warning-insight"
            st.markdown(f"""
            <div class="{advice_color}">
            <h4>Current Rate: {savings_advice['current_rate']:.1f}%</h4>
            <p>{savings_advice['advice']}</p>
            <p><strong>Target:</strong> {savings_advice['target_rate']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎯 Action Plan")
            action_plan = ai_profile['action_plan']
            
            for action in action_plan[:3]:  # Show top 3 actions
                priority_color = "#dc143c" if action['priority'] == 'High' else "#ffa500" if action['priority'] == 'Medium' else "#32cd32"
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-left: 4px solid {priority_color}; 
                            margin: 0.5rem 0; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h5 style="color: {priority_color}; margin: 0;">{action['timeframe']}</h5>
                <h4 style="margin: 0.5rem 0;">{action['action']}</h4>
                <p style="margin: 0; color: #666;">{action['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Emergency Fund Status
        st.markdown("### 🚨 Emergency Fund Analysis")
        emergency_status = ai_profile['emergency_fund_status']
        
        status_colors = {
            "Excellent": "#2e8b57",
            "Good": "#32cd32", 
            "Fair": "#ffa500",
            "Poor": "#dc143c"
        }
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: {status_colors.get(emergency_status['status'], '#666')}; 
                        border-radius: 1rem; color: white;">
            <h3>Emergency Fund</h3>
            <h2>{emergency_status['status']}</h2>
            <p>{emergency_status['current_months']:.1f} months covered</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding: 1.5rem; background: #f8f9fa; border-radius: 1rem; margin-top: 0.5rem;">
            <p><strong>Recommendation:</strong> {emergency_status['advice']}</p>
            <p><strong>Target:</strong> {emergency_status['recommended_months']} months of expenses</p>
            <p><strong>Shortfall:</strong> {analyzer.symbol}{emergency_status['shortfall']:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating AI insights: {e}")

def create_data_generator_interface():
    """Interface for generating new synthetic data"""
    
    st.subheader("🏭 Data Generator")
    st.markdown("Generate realistic financial data for testing and demo purposes.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        country = st.selectbox("Country", ["UK", "India"])
    
    with col2:
        profile = st.selectbox("Profile", ["student", "medium_spender", "family", "high_earner"])
    
    with col3:
        year = st.selectbox("Year", [2024, 2023])
    
    if st.button("🚀 Generate New Dataset", type="primary"):
        with st.spinner("Generating realistic financial data..."):
            try:
                generator = FinanceDataGenerator(country)
                data = generator.generate_year_data(year, profile)
                filename = f"{country.lower()}_{profile}_{year}.csv"
                filepath = generator.save_data(data, filename)
                
                st.success(f"✅ Generated {len(data)} transactions!")
                st.info(f"📁 Saved to: {filepath}")
                
                # Show preview
                st.markdown("### 👀 Data Preview")
                st.dataframe(data.head(10))
                
                # Offer to load this data
                if st.button("📊 Analyze This Data"):
                    analyzer = FinancialAnalyzer(dataframe=data)
                    st.session_state.analyzer = analyzer
                    st.session_state.current_data_source = f"Generated: {country} {profile} {year}"
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error generating data: {e}")

def main():
    """Main application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header with animation effect
    st.markdown('<h1 class="main-header">💰 Smart Personal Finance Assistant Pro</h1>', unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.title("🎛️ Navigation")
    
    # Data source selection
    st.sidebar.header("📊 Data Source")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["📁 Upload CSV File", "🎯 Use Sample Data", "🏭 Generate New Data"]
    )
    
    # Main content area
    if data_option == "📁 Upload CSV File":
        st.sidebar.markdown("### Upload Your Data")
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload your bank statement CSV file"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_columns = ['date', 'amount', 'category', 'merchant']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                    st.info("Required columns: date, amount, category, merchant, description")
                else:
                    analyzer = FinancialAnalyzer(dataframe=df)
                    ml_predictor = MLFinancialPredictor(dataframe=df)
                    
                    st.session_state.analyzer = analyzer
                    st.session_state.ml_predictor = ml_predictor
                    st.session_state.current_data_source = f"Uploaded: {uploaded_file.name}"
                    
                    st.sidebar.success("✅ File uploaded successfully!")
                    
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
    
    elif data_option == "🎯 Use Sample Data":
        available_files = load_sample_data()
        
        if available_files:
            selected_file = st.sidebar.selectbox(
                "Choose sample dataset:",
                list(available_files.keys())
            )
            
            if st.sidebar.button("📊 Load Sample Data", type="primary"):
                try:
                    analyzer = FinancialAnalyzer(data_file=available_files[selected_file])
                    ml_predictor = MLFinancialPredictor(data_file=available_files[selected_file])
                    
                    st.session_state.analyzer = analyzer
                    st.session_state.ml_predictor = ml_predictor
                    st.session_state.current_data_source = selected_file
                    
                    st.sidebar.success("✅ Sample data loaded!")
                    st.rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"Error loading sample data: {e}")
        else:
            st.sidebar.warning("⚠️ No sample data found!")
            st.sidebar.code("python src/data_generator.py", language="bash")
    
    else:  # Generate New Data
        st.sidebar.markdown("### 🏭 Generate Data")
        if st.sidebar.button("🚀 Go to Generator"):
            st.session_state.show_generator = True
    
    # Main dashboard
    if st.session_state.analyzer is not None:
        # Current data source info
        st.info(f"📊 Currently analyzing: **{st.session_state.current_data_source}**")
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", "🤖 AI Insights", "📈 ML Predictions", "📋 Reports", "⚙️ Settings"
        ])
        
        with tab1:
            st.header("📊 Financial Dashboard")
            create_enhanced_metrics(st.session_state.analyzer)
            
            st.markdown("---")
            create_advanced_charts(st.session_state.analyzer)
        
        with tab2:
            st.header("🤖 AI Financial Advisor")
            display_ai_insights(st.session_state.analyzer)
        
        with tab3:
            st.header("📈 Machine Learning Predictions")
            
            if st.session_state.ml_predictor is not None:
                # Train models if needed
                if st.button("🧠 Train ML Models"):
                    with st.spinner("Training machine learning models..."):
                        try:
                            expense_results = st.session_state.ml_predictor.train_expense_predictor()
                            anomaly_results = st.session_state.ml_predictor.train_anomaly_detector()
                            
                            if expense_results:
                                st.success(f"✅ Expense Predictor trained! R² Score: {expense_results['r2_score']:.3f}")
                            
                            if anomaly_results:
                                st.success(f"✅ Anomaly Detector trained! Found {anomaly_results['anomalies_detected']} anomalies")
                        except Exception as e:
                            st.error(f"Error training models: {e}")
                
                # Next month prediction
                st.subheader("🔮 Next Month Prediction")
                try:
                    prediction = st.session_state.ml_predictor.predict_next_month_expenses()
                    
                    if prediction and 'error' not in prediction:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Predicted Expenses",
                                f"{st.session_state.analyzer.symbol}{prediction['total_predicted_expenses']:,.2f}",
                                delta=f"Confidence: {prediction['confidence']}"
                            )
                        
                        with col2:
                            # Category breakdown
                            category_pred = prediction['category_breakdown']
                            if category_pred:
                                pred_df = pd.DataFrame.from_dict(category_pred, orient='index')
                                fig_pred = px.bar(
                                    y=pred_df.index,
                                    x=pred_df['predicted_monthly_total'],
                                    orientation='h',
                                    title="Predicted Expenses by Category"
                                )
                                st.plotly_chart(fig_pred, use_container_width=True)
                    else:
                        st.info("Train models first to enable predictions!")
                except Exception as e:
                    st.error(f"Error generating predictions: {e}")
                
                # Anomaly report
                st.subheader("🚨 Anomaly Detection")
                try:
                    anomaly_report = st.session_state.ml_predictor.get_anomaly_report()
                    
                    if anomaly_report and 'message' in anomaly_report:
                        st.success(anomaly_report['message'])
                    elif anomaly_report:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Anomalies", anomaly_report.get('total_anomalies', 0))
                            st.metric("Anomaly Rate", f"{anomaly_report.get('anomaly_percentage', 0):.1f}%")
                        
                        with col2:
                            most_unusual = anomaly_report.get('most_unusual_transaction', {})
                            if most_unusual:
                                st.markdown(f"""
                                **Most Unusual Transaction:**
                                - Amount: {st.session_state.analyzer.symbol}{most_unusual.get('amount', 0):,.2f}
                                - Date: {most_unusual.get('date', 'N/A')}
                                - Merchant: {most_unusual.get('merchant', 'N/A')}
                                - Category: {most_unusual.get('category', 'N/A')}
                                """)
                    else:
                        st.info("Train anomaly detector first to enable anomaly detection!")
                except Exception as e:
                    st.error(f"Error generating anomaly report: {e}")
            else:
                st.info("Load data to enable ML predictions!")
        
        with tab4:
            st.header("📋 Detailed Reports")
            
            # Monthly report selector
            col1, col2, col3 = st.columns(3)
            
            with col1:
                report_year = st.selectbox("Year", [2024, 2023])
            with col2:
                report_month = st.selectbox("Month", range(1, 13), index=7, 
                                          format_func=lambda x: datetime(2024, x, 1).strftime('%B'))
            with col3:
                if st.button("📊 Generate Report"):
                    try:
                        report = st.session_state.analyzer.generate_monthly_report(report_year, report_month)
                        st.text(report)
                    except Exception as e:
                        st.error(f"Error generating report: {e}")
            
            # Export options
            st.subheader("📤 Export Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export Analysis"):
                    try:
                        # Create analysis summary
                        stats = st.session_state.analyzer.get_overview_stats()
                        category_analysis = st.session_state.analyzer.analyze_spending_by_category()
                        
                        export_data = {
                            'overview': stats,
                            'categories': category_analysis.to_dict(),
                            'export_date': datetime.now().isoformat()
                        }
                        
                        st.download_button(
                            "💾 Download JSON",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Error creating export: {e}")
            
            with col2:
                if st.session_state.ml_predictor and st.button("🤖 Export ML Models"):
                    try:
                        st.session_state.ml_predictor.save_models()
                        st.success("✅ Models saved to 'models' folder!")
                    except Exception as e:
                        st.error(f"Error saving models: {e}")
        
        with tab5:
            st.header("⚙️ Settings & Configuration")
            
            # Theme selection
            st.subheader("🎨 Appearance")
            theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2"])
            
            # Data settings
            st.subheader("📊 Data Settings")
            show_raw_data = st.checkbox("Show Raw Transaction Data")
            
            if show_raw_data:
                st.subheader("📋 Raw Transaction Data")
                st.dataframe(st.session_state.analyzer.df)
            
            # Advanced settings
            st.subheader("🔧 Advanced Settings")
            anomaly_threshold = st.slider("Anomaly Detection Sensitivity", 0.05, 0.3, 0.1)
            prediction_months = st.slider("Prediction Horizon (months)", 1, 12, 6)
    
    elif hasattr(st.session_state, 'show_generator') and st.session_state.show_generator:
        create_data_generator_interface()
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
        <h2>🌟 Welcome to Smart Finance Assistant Pro!</h2>
        <p style="font-size: 1.2rem; color: #666;">Your AI-powered financial analysis companion</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        
        features = [
            ("📊", "Interactive Analytics", "Beautiful charts and deep financial insights"),
            ("🤖", "AI Financial Advisor", "Personalized recommendations and advice"),  
            ("📈", "ML Predictions", "Forecast future spending with machine learning"),
            ("🔍", "Anomaly Detection", "Automatically spot unusual transactions"),
            ("🌍", "Multi-Currency", "Support for UK (£) and India (₹)"),
            ("📱", "Mobile Friendly", "Works perfectly on all devices")
        ]
        
        for i, (icon, title, desc) in enumerate(features):
            col = [col1, col2, col3][i % 3]
            
            with col:
                st.markdown(f"""
                <div class="feature-card">
                <h2 style="font-size: 3rem; margin-bottom: 1rem;">{icon}</h2>
                <h3 style="color: #1f77b4; margin-bottom: 1rem;">{title}</h3>
                <p style="color: #666;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Getting started
        st.markdown("---")
        st.markdown("""
        ### 🚀 Getting Started:
        1. **📁 Upload your CSV** - Use your bank statement export
        2. **🎯 Try sample data** - Explore with realistic demo data  
        3. **🏭 Generate data** - Create custom synthetic datasets
        
        ### 📋 CSV Format Required:
        Your CSV should have these columns: `date`, `amount`, `category`, `merchant`, `description`
        
        ### 💡 Pro Tips:
        - Export your bank statements as CSV files for real analysis
        - Try different profiles in the data generator 
        - Use AI insights to improve your financial health!
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ using Python, Streamlit, Plotly & AI</p>
    <p>🎓 <strong>Perfect for:</strong> Portfolio projects, Financial learning, Personal finance tracking</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()