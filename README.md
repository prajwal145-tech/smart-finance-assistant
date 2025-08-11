# Smart Personal Finance Assistant

This is a simple yet powerful personal finance analysis tool I built using Python, Streamlit, and machine learning.  
It takes your transaction data, analyzes spending patterns, predicts future expenses, and even flags unusual activity.  
The goal is to make sense of day-to-day spending in a way that's actually useful.

## What it does
- Upload your CSV file with transactions and get a clean breakdown of your spending.
- Predict upcoming expenses using past patterns (Random Forest regression).
- Detect transactions that look unusual compared to your normal habits.
- Highlight seasonal trends (months you usually spend more/less).
- Show weekend vs weekday spending differences.
- Identify the merchants you use the most.

## Why I built this
I wanted a lightweight, local-first finance assistant that wasn’t tied to any bank’s ecosystem or a third-party app.  
Everything runs on your machine. You control your data.

## Tech used
- **Streamlit** – for the interactive web app
- **Pandas & NumPy** – for handling and cleaning the data
- **Scikit-Learn** – for machine learning models
- **Matplotlib** – for basic charts
- **Joblib** – for saving/loading trained models

## How to run it

### Clone the repo
```bash
git clone https://github.com/yourusername/smart-finance-assistant.git
cd smart-finance-assistant
