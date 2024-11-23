import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import json

def create_figure_directories():
    """Create standardized directory structure for all predictions"""
    base_dir = 'figures'
    directories = {
        'customer_segmentation': ['charts', 'data'],
        'product_forecast': {
            'daily__forecast': ['charts', 'data'],
            'weekly_forecast': ['charts', 'data'],
            'monthly_forecast': ['charts', 'data']
        },
        'sales_forecast': ['charts', 'data']
    }

    for main_dir, subdirs in directories.items():
        if isinstance(subdirs, list):
            for subdir in subdirs:
                os.makedirs(os.path.join(base_dir, main_dir, subdir), exist_ok=True)
        else:
            for forecast_type, forecast_subdirs in subdirs.items():
                for subdir in forecast_subdirs:
                    os.makedirs(os.path.join(base_dir, main_dir, forecast_type, subdir), exist_ok=True)

    return base_dir

def load_data():
    #Load environment variables
    load_dotenv()

    # Get credentials from environment variables
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASS')

    # Create a database connection
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    

    #SQL Query
    query = """
    SELECT
        c.CustomerID,
        SUM(s.total_amount) AS total_sales,
        COUNT(s.sale_id) AS order_frequency
    FROM
        Dim_Customer c
    JOIN
        Fact_Sales s ON c.customer_key = s.customer_key
    GROUP BY
        c.CustomerID
    """

    return pd.read_sql(query, engine)

def process_data(df):
    #Load model and scaler