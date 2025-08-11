"""
Smart Personal Finance Assistant - ML Financial Predictor
=======================================================

This module uses machine learning to predict future spending patterns,
detect anomalies, and provide data-driven financial insights.

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class MLFinancialPredictor:
    """
    Machine Learning-powered financial predictor and analyzer
    """
    
    def __init__(self, data_file=None, dataframe=None):
        """
        Initialize the ML predictor
        
        Args:
            data_file (str): Path to CSV file with transactions
            dataframe (pd.DataFrame): DataFrame with transaction data
        """
        self.label_encoder = LabelEncoder()
        
        if data_file:
            self.df = pd.read_csv(data_file)
        elif dataframe is not None:
            self.df = dataframe.copy()
        else:
            raise ValueError("Please provide either data_file or dataframe")
        
        # Prepare data for ML
        self.prepare_ml_features()
        
        # Initialize models
        self.expense_predictor = None
        self.anomaly_detector = None
        self.category_predictor = None
        
        # Scalers and encoders
        self.scaler = StandardScaler()
        
        print(f"🤖 ML Financial Predictor initialized with {len(self.df)} transactions")
    
    def prepare_ml_features(self):
        """Prepare features for machine learning models"""
        
        # Convert date and create time-based features
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_month'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])
        self.df['is_month_start'] = self.df['day_of_month'] <= 5
        self.df['is_month_end'] = self.df['day_of_month'] >= 25
        
        # Amount features
        self.df['abs_amount'] = abs(self.df['amount'])
        self.df['is_expense'] = self.df['amount'] < 0
        self.df['is_income'] = self.df['amount'] > 0
        
        # Category encoding for ML
        self.df['category_encoded'] = self.label_encoder.fit_transform(self.df['category'])
        
        # Rolling statistics (spending patterns)
        self.df = self.df.sort_values('date')
        self.df['rolling_mean_7d'] = self.df['abs_amount'].rolling(window=7, min_periods=1).mean()
        self.df['rolling_mean_30d'] = self.df['abs_amount'].rolling(window=30, min_periods=1).mean()
        self.df['days_since_last_transaction'] = self.df['date'].diff().dt.days.fillna(0)
        
        # Seasonal features
        self.df['is_holiday_season'] = self.df['month'].isin([11, 12, 1])  # Nov, Dec, Jan
        self.df['is_summer'] = self.df['month'].isin([6, 7, 8])
        
        # Merchant frequency (how often they shop at each merchant)
        merchant_counts = self.df['merchant'].value_counts()
        self.df['merchant_frequency'] = self.df['merchant'].map(merchant_counts)
        
        print("✅ ML features prepared successfully!")
    
    def predict_monthly_expenses(self, age: int, income: float, family_size: int, location_type: str = 'urban') -> dict:
        """Predict monthly expenses based on user profile"""
        # Rule-based prediction
        base_expenses = income * 0.7 / 12  # 70% of income as expenses
        family_multiplier = 1 + (family_size - 1) * 0.15
        location_multiplier = {'urban': 1.2, 'suburban': 1.0, 'rural': 0.8}.get(location_type, 1.0)
        prediction = base_expenses * family_multiplier * location_multiplier
        
        # Ensure reasonable bounds
        prediction = max(1000, min(prediction, income * 0.9 / 12))
        
        return {
            'predicted_monthly_expenses': prediction,
            'confidence_level': 0.75,
            'category_breakdown': self._estimate_expense_breakdown(prediction),
            'model_used': 'Rule-based'
        }
    
    def _estimate_expense_breakdown(self, total_expenses: float) -> dict:
        """Estimate expense breakdown by category"""
        return {
            'Housing': total_expenses * 0.35,
            'Food': total_expenses * 0.15,
            'Transportation': total_expenses * 0.15,
            'Utilities': total_expenses * 0.08,
            'Healthcare': total_expenses * 0.08,
            'Entertainment': total_expenses * 0.07,
            'Clothing': total_expenses * 0.05,
            'Other': total_expenses * 0.07
        }
    
    def train_expense_predictor(self):
        """Train a model to predict future expenses"""
        
        # Prepare data for expense prediction
        expense_data = self.df[self.df['is_expense']].copy()
        
        if len(expense_data) < 50:
            print("⚠️ Not enough expense data to train reliable model")
            return {'mae': 0, 'r2_score': 0, 'feature_importance': pd.DataFrame()}
        
        # Features for prediction
        feature_columns = [
            'month', 'day_of_week', 'day_of_month', 'category_encoded',
            'is_weekend', 'is_month_start', 'is_month_end', 'is_holiday_season',
            'rolling_mean_7d', 'rolling_mean_30d', 'merchant_frequency'
        ]
        
        X = expense_data[feature_columns].fillna(0)
        y = expense_data['abs_amount']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.expense_predictor = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        self.expense_predictor.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.expense_predictor.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Expense Predictor trained!")
        print(f"   Mean Absolute Error: {mae:.2f}")
        print(f"   R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.expense_predictor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'mae': mae,
            'r2_score': r2,
            'feature_importance': feature_importance
        }
    
    def train_anomaly_detector(self):
        """Train anomaly detection model to find unusual transactions"""
        
        # Features for anomaly detection
        anomaly_features = [
            'abs_amount', 'day_of_week', 'month', 'category_encoded',
            'is_weekend', 'merchant_frequency', 'rolling_mean_30d'
        ]
        
        X_anomaly = self.df[anomaly_features].fillna(0)
        
        # Train Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # Expect 10% of transactions to be anomalies
            random_state=42
        )
        
        self.anomaly_detector.fit(X_anomaly)
        
        # Predict anomalies
        anomaly_scores = self.anomaly_detector.decision_function(X_anomaly)
        anomaly_predictions = self.anomaly_detector.predict(X_anomaly)
        
        # Add results to dataframe
        self.df['anomaly_score'] = anomaly_scores
        self.df['is_anomaly'] = anomaly_predictions == -1
        
        anomaly_count = self.df['is_anomaly'].sum()
        print(f"✅ Anomaly Detector trained!")
        print(f"   Detected {anomaly_count} potential anomalies ({anomaly_count/len(self.df)*100:.1f}%)")
        
        return {
            'anomalies_detected': anomaly_count,
            'anomaly_percentage': anomaly_count/len(self.df)*100
        }
    
    def predict_next_month_expenses(self):
        """
        Predict expenses for the next month based on historical patterns
        
        Returns:
            dict: Prediction results including total amount and category breakdown
        """
        try:
            # Check if expense predictor is trained
            if self.expense_predictor is None:
                # Use historical average as fallback
                return self._get_historical_prediction()
            
            # Get the most recent month for feature reference
            latest_date = self.df['date'].max()
            next_month_date = latest_date + timedelta(days=30)
            
            # Create features for next month prediction
            next_month_features = {
                'month': next_month_date.month,
                'day_of_week': 1,  # Average weekday
                'day_of_month': 15,  # Mid-month
                'category_encoded': 0,  # Will predict for each category
                'is_weekend': False,
                'is_month_start': False,
                'is_month_end': False,
                'is_holiday_season': next_month_date.month in [11, 12, 1],
                'rolling_mean_7d': self.df['rolling_mean_7d'].tail(30).mean(),
                'rolling_mean_30d': self.df['rolling_mean_30d'].tail(30).mean(),
                'merchant_frequency': self.df['merchant_frequency'].mean()
            }
            
            # Predict for each category
            category_predictions = {}
            total_predicted = 0
            
            # Get unique categories and their encoded values
            unique_categories = self.df['category'].unique()
            
            for category in unique_categories:
                # Create feature vector for this category
                category_encoded = self.label_encoder.transform([category])[0]
                features = next_month_features.copy()
                features['category_encoded'] = category_encoded
                
                # Convert to array and scale
                feature_array = np.array([list(features.values())])
                feature_array_scaled = self.scaler.transform(feature_array)
                
                # Predict
                category_prediction = self.expense_predictor.predict(feature_array_scaled)[0]
                
                # Count transactions per category to estimate monthly total
                category_transactions = self.df[self.df['category'] == category]
                monthly_transaction_count = len(category_transactions) / len(self.df['month'].unique())
                
                predicted_monthly_total = category_prediction * max(1, monthly_transaction_count)
                
                category_predictions[category] = {
                    'predicted_per_transaction': category_prediction,
                    'predicted_monthly_total': predicted_monthly_total,
                    'historical_avg_transactions': monthly_transaction_count
                }
                
                total_predicted += predicted_monthly_total
            
            # Calculate confidence based on model performance
            confidence = "Medium" if hasattr(self, '_last_r2_score') and self._last_r2_score > 0.5 else "Low"
            
            return {
                'total_predicted_expenses': total_predicted,
                'confidence': confidence,
                'category_breakdown': category_predictions,
                'prediction_date': next_month_date.strftime('%Y-%m'),
                'model_used': 'Machine Learning'
            }
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return self._get_historical_prediction()
    
    def _get_historical_prediction(self):
        """Fallback prediction based on historical averages"""
        try:
            # Calculate monthly averages
            expense_data = self.df[self.df['is_expense']].copy()
            
            # Group by month and calculate averages
            monthly_expenses = expense_data.groupby(['year', 'month'])['abs_amount'].sum()
            avg_monthly_expenses = monthly_expenses.mean()
            
            # Category breakdown
            category_averages = expense_data.groupby('category')['abs_amount'].sum()
            monthly_category_avg = category_averages / len(monthly_expenses)
            
            category_breakdown = {}
            for category, amount in monthly_category_avg.items():
                category_breakdown[category] = {
                    'predicted_per_transaction': amount / max(1, len(expense_data[expense_data['category'] == category])),
                    'predicted_monthly_total': amount,
                    'historical_avg_transactions': len(expense_data[expense_data['category'] == category]) / len(monthly_expenses)
                }
            
            return {
                'total_predicted_expenses': avg_monthly_expenses,
                'confidence': 'Medium (Historical)',
                'category_breakdown': category_breakdown,
                'prediction_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m'),
                'model_used': 'Historical Average'
            }
            
        except Exception as e:
            print(f"Error in historical prediction: {e}")
            return {
                'error': f"Could not generate prediction: {e}",
                'total_predicted_expenses': 0,
                'confidence': 'Low',
                'category_breakdown': {},
                'model_used': 'Error Fallback'
            }
    
    def get_anomaly_report(self):
        """
        Get a comprehensive anomaly detection report
        
        Returns:
            dict: Anomaly report with statistics and examples
        """
        try:
            # Check if anomaly detector is trained
            if self.anomaly_detector is None or 'is_anomaly' not in self.df.columns:
                return {'message': 'Train the anomaly detector first to see anomaly reports!'}
            
            # Get anomalies
            anomalies = self.df[self.df['is_anomaly']].copy()
            
            if len(anomalies) == 0:
                return {'message': 'No anomalies detected! All transactions appear normal.'}
            
            # Sort anomalies by score (most unusual first)
            anomalies = anomalies.sort_values('anomaly_score')
            
            # Find the most unusual transaction
            most_unusual = anomalies.iloc[0] if len(anomalies) > 0 else None
            
            # Anomaly statistics by category
            anomaly_by_category = anomalies.groupby('category').agg({
                'abs_amount': ['count', 'mean', 'sum'],
                'anomaly_score': 'mean'
            }).round(2)
            
            # Recent anomalies (last 30 days)
            recent_cutoff = datetime.now() - timedelta(days=30)
            recent_anomalies = anomalies[anomalies['date'] >= recent_cutoff]
            
            # Top merchants with anomalies
            merchant_anomalies = anomalies.groupby('merchant').agg({
                'abs_amount': ['count', 'sum']
            }).round(2)
            top_anomaly_merchants = merchant_anomalies.sort_values(('abs_amount', 'count'), ascending=False).head(3)
            
            report = {
                'total_anomalies': len(anomalies),
                'anomaly_percentage': round((len(anomalies) / len(self.df)) * 100, 1),
                'recent_anomalies': len(recent_anomalies),
                'most_unusual_transaction': {
                    'amount': most_unusual['abs_amount'],
                    'date': most_unusual['date'].strftime('%Y-%m-%d'),
                    'merchant': most_unusual['merchant'],
                    'category': most_unusual['category'],
                    'description': most_unusual.get('description', 'N/A'),
                    'anomaly_score': round(most_unusual['anomaly_score'], 3)
                } if most_unusual is not None else {},
                'category_breakdown': anomaly_by_category.to_dict() if not anomaly_by_category.empty else {},
                'top_anomaly_merchants': top_anomaly_merchants.to_dict() if not top_anomaly_merchants.empty else {},
                'anomaly_summary': {
                    'highest_amount': anomalies['abs_amount'].max() if not anomalies.empty else 0,
                    'lowest_amount': anomalies['abs_amount'].min() if not anomalies.empty else 0,
                    'average_amount': anomalies['abs_amount'].mean() if not anomalies.empty else 0,
                    'total_anomaly_amount': anomalies['abs_amount'].sum() if not anomalies.empty else 0
                }
            }
            
            return report
            
        except Exception as e:
            return {'error': f"Could not generate anomaly report: {e}"}
    
    def get_spending_insights_ml(self):
        """
        Get ML-powered spending insights
        
        Returns:
            list: List of insight dictionaries
        """
        insights = []
        
        try:
            # Insight 1: Spending trend analysis
            monthly_spending = self.df[self.df['is_expense']].groupby(['year', 'month'])['abs_amount'].sum()
            
            if len(monthly_spending) >= 3:
                recent_trend = monthly_spending.tail(3).pct_change().mean()
                
                if recent_trend > 0.1:
                    insights.append({
                        'title': 'Increasing Spending Trend',
                        'description': f'Your spending has been increasing by an average of {recent_trend*100:.1f}% per month recently. Consider reviewing your budget.',
                        'impact': 'high',
                        'category': 'trend'
                    })
                elif recent_trend < -0.1:
                    insights.append({
                        'title': 'Decreasing Spending Trend',
                        'description': f'Great job! Your spending has been decreasing by {abs(recent_trend)*100:.1f}% per month on average.',
                        'impact': 'positive',
                        'category': 'trend'
                    })
            
            # Insight 2: Category concentration risk
            category_spending = self.df[self.df['is_expense']].groupby('category')['abs_amount'].sum()
            total_expenses = category_spending.sum()
            
            if len(category_spending) > 0:
                max_category = category_spending.idxmax()
                max_percentage = (category_spending.max() / total_expenses) * 100
                
                if max_percentage > 40:
                    insights.append({
                        'title': f'High Concentration in {max_category}',
                        'description': f'{max_percentage:.1f}% of your spending goes to {max_category}. Consider diversifying your expenses for better financial balance.',
                        'impact': 'medium',
                        'category': 'distribution'
                    })
            
            # Insight 3: Weekend vs weekday spending
            weekday_spending = self.df[(self.df['is_expense']) & (~self.df['is_weekend'])]['abs_amount'].mean()
            weekend_spending = self.df[(self.df['is_expense']) & (self.df['is_weekend'])]['abs_amount'].mean()
            
            if weekend_spending > weekday_spending * 1.5:
                insights.append({
                    'title': 'High Weekend Spending',
                    'description': f'You spend {(weekend_spending/weekday_spending-1)*100:.1f}% more on weekends. Consider planning weekend activities with a budget.',
                    'impact': 'medium',
                    'category': 'behavior'
                })
            
            # Insight 4: Merchant frequency analysis
            merchant_counts = self.df['merchant'].value_counts()
            top_merchant = merchant_counts.index[0] if len(merchant_counts) > 0 else None
            top_merchant_percentage = (merchant_counts.iloc[0] / len(self.df)) * 100 if len(merchant_counts) > 0 else 0
            
            if top_merchant and top_merchant_percentage > 20:
                insights.append({
                    'title': f'Frequent Transactions at {top_merchant}',
                    'description': f'{top_merchant_percentage:.1f}% of your transactions are with {top_merchant}. This could indicate a regular habit or subscription.',
                    'impact': 'low',
                    'category': 'pattern'
                })
            
            # Add default insight if none found
            if not insights:
                insights.append({
                    'title': 'Healthy Spending Pattern',
                    'description': 'Your spending patterns look balanced and healthy. Keep up the good financial habits!',
                    'impact': 'positive',
                    'category': 'overall'
                })
            
        except Exception as e:
            insights.append({
                'title': 'Analysis Error',
                'description': f'Could not analyze spending patterns: {e}',
                'impact': 'low',
                'category': 'error'
            })
        
        return insights
    
    def create_spending_forecast_chart(self):
        """
        Create data for spending forecast visualization
        
        Returns:
            pd.DataFrame: Forecast data for charting
        """
        try:
            # Generate forecast for next 6 months
            forecast_data = []
            
            # Get historical monthly averages
            expense_data = self.df[self.df['is_expense']].copy()
            monthly_avg = expense_data.groupby(['year', 'month'])['abs_amount'].sum().mean()
            
            # Create forecast with some variation
            base_date = datetime.now()
            
            for i in range(6):
                forecast_month = base_date + timedelta(days=30 * i)
                
                # Add some seasonal variation
                seasonal_factor = 1.0
                if forecast_month.month in [11, 12]:  # Holiday season
                    seasonal_factor = 1.2
                elif forecast_month.month in [6, 7, 8]:  # Summer
                    seasonal_factor = 1.1
                
                # Add some random variation (±10%)
                variation = 1 + (np.random.random() - 0.5) * 0.2
                
                predicted_amount = monthly_avg * seasonal_factor * variation
                
                forecast_data.append({
                    'month': forecast_month.strftime('%Y-%m'),
                    'predicted_expenses': predicted_amount
                })
            
            return pd.DataFrame(forecast_data)
            
        except Exception as e:
            print(f"Error creating forecast: {e}")
            return pd.DataFrame()
    
    def save_models(self, models_dir='models'):
        """Save trained models to disk"""
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        try:
            if self.expense_predictor is not None:
                joblib.dump(self.expense_predictor, f'{models_dir}/expense_predictor.pkl')
                print("✅ Expense predictor saved!")
            
            if self.anomaly_detector is not None:
                joblib.dump(self.anomaly_detector, f'{models_dir}/anomaly_detector.pkl')
                print("✅ Anomaly detector saved!")
            
            # Save scalers and encoders
            joblib.dump(self.scaler, f'{models_dir}/scaler.pkl')
            joblib.dump(self.label_encoder, f'{models_dir}/label_encoder.pkl')
            print("✅ Scalers and encoders saved!")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def load_models(self, models_dir='models'):
        """Load trained models from disk"""
        
        try:
            if os.path.exists(f'{models_dir}/expense_predictor.pkl'):
                self.expense_predictor = joblib.load(f'{models_dir}/expense_predictor.pkl')
                print("✅ Expense predictor loaded!")
            
            if os.path.exists(f'{models_dir}/anomaly_detector.pkl'):
                self.anomaly_detector = joblib.load(f'{models_dir}/anomaly_detector.pkl')
                print("✅ Anomaly detector loaded!")
            
            # Load scalers and encoders
            if os.path.exists(f'{models_dir}/scaler.pkl'):
                self.scaler = joblib.load(f'{models_dir}/scaler.pkl')
            
            if os.path.exists(f'{models_dir}/label_encoder.pkl'):
                self.label_encoder = joblib.load(f'{models_dir}/label_encoder.pkl')
            
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")