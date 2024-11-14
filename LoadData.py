import pandas as pd
from sqlalchemy import create_engine, text

# Load the .xls file
file_path = './cleaned_retail_data.csv'

# Add error handling for file loading
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# Add data validation
if data.empty:
    print("Error: Dataset is empty")
    exit(1)

# Extract unique products and customers
unique_products = data[['StockCode', 'Description']].drop_duplicates()
unique_customers = data[data['CustomerID'].notna()][['CustomerID', 'Country']].drop_duplicates()

# Create time dimension first
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], dayfirst=True)
data['year'] = data['InvoiceDate'].dt.year
data['month'] = data['InvoiceDate'].dt.month
data['day'] = data['InvoiceDate'].dt.day
data['quarter'] = data['InvoiceDate'].dt.quarter

dim_time = data[['InvoiceDate', 'year', 'month', 'day', 'quarter']].drop_duplicates().rename(columns={'InvoiceDate': 'date'})

# Connect to PostgreSQL and load data
try:
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/Retail_Store_Datawarehouse')
    
    # Drop tables in correct order
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS fact_sales CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS dim_product CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS dim_customer CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS dim_time CASCADE"))
        conn.commit()
    
    # Load dimension tables first
    unique_products.to_sql('dim_product', engine, if_exists='replace', index=False)
    unique_customers.to_sql('dim_customer', engine, if_exists='replace', index=False)
    dim_time.to_sql('dim_time', engine, if_exists='replace', index=False)
    
    # Create fact_sales
    fact_sales = data.merge(unique_products, on='StockCode').merge(unique_customers, on='CustomerID')
    fact_sales = fact_sales.merge(dim_time, left_on='InvoiceDate', right_on='date')
    fact_sales['total_amount'] = fact_sales['Quantity'] * fact_sales['UnitPrice']
    fact_sales = fact_sales[['InvoiceNo', 'Quantity', 'UnitPrice', 'total_amount', 'StockCode', 'CustomerID', 'date']]
    
    # Load fact table
    fact_sales.to_sql('fact_sales', engine, if_exists='replace', index=False)

except Exception as e:
    print(f"Database error: {str(e)}")
    exit(1)
