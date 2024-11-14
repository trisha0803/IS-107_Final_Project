import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar

# Set page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv('../cleaned_retail_data.csv')
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], dayfirst=True)
    data['Revenue'] = data['Quantity'] * data['UnitPrice']
    return data

data = load_data()

# Sidebar filters
st.sidebar.header('Filters')

# Date range filter
min_date = data['InvoiceDate'].min().date()
max_date = data['InvoiceDate'].max().date()

start_date = st.sidebar.date_input('Start Date', min_date)
end_date = st.sidebar.date_input('End Date', max_date)

# Country filter
countries = ['All'] + list(data['Country'].unique())
selected_country = st.sidebar.selectbox('Select Country', countries)

# Filter data based on selections
mask = (data['InvoiceDate'].dt.date >= start_date) & (data['InvoiceDate'].dt.date <= end_date)
if selected_country != 'All':
    mask = mask & (data['Country'] == selected_country)
filtered_data = data[mask]

# Main dashboard
st.title('📊 Retail Analytics Dashboard')

# Top metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = filtered_data['Revenue'].sum()
    st.metric("Total Revenue", f"₱{total_revenue:,.2f}")

with col2:
    total_orders = filtered_data['InvoiceNo'].nunique()
    st.metric("Total Orders", f"{total_orders:,}")

with col3:
    total_customers = filtered_data['CustomerID'].nunique()
    st.metric("Total Customers", f"{total_customers:,}")

with col4:
    avg_order_value = total_revenue / total_orders
    st.metric("Average Order Value", f"₱{avg_order_value:.2f}")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Daily Revenue Trend")
    daily_revenue = filtered_data.groupby(filtered_data['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
    fig = px.line(daily_revenue, x='InvoiceDate', y='Revenue',
                  title='Daily Revenue Over Time')
    fig.update_layout(xaxis_title="Date", yaxis_title="Revenue (£)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Revenue by Country")
    country_revenue = filtered_data.groupby('Country')['Revenue'].sum().sort_values(ascending=True)
    fig = px.bar(country_revenue, orientation='h',
                 title='Revenue by Country')
    fig.update_layout(xaxis_title="Revenue (£)", yaxis_title="Country")
    st.plotly_chart(fig, use_container_width=True)

# Second row of charts
col3, col4 = st.columns(2)

with col3:
    st.subheader("Top 10 Products by Revenue")
    product_revenue = filtered_data.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(product_revenue, orientation='h',
                 title='Top 10 Products by Revenue')
    fig.update_layout(xaxis_title="Revenue (£)", yaxis_title="Product")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Monthly Revenue Distribution")
    filtered_data['Month'] = filtered_data['InvoiceDate'].dt.strftime('%B %Y')
    monthly_revenue = filtered_data.groupby('Month')['Revenue'].sum().reset_index()
    fig = px.bar(monthly_revenue, x='Month', y='Revenue',
                 title='Monthly Revenue Distribution')
    fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (£)")
    st.plotly_chart(fig, use_container_width=True)

# Additional Analysis
st.subheader("Detailed Analysis")
tabs = st.tabs(["Customer Analysis", "Product Analysis", "Time Analysis"])

with tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer purchase frequency
        purchase_frequency = filtered_data.groupby('CustomerID').size().value_counts()
        fig = px.bar(purchase_frequency, title='Customer Purchase Frequency Distribution')
        fig.update_layout(xaxis_title="Number of Orders", yaxis_title="Number of Customers")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer revenue distribution
        customer_revenue = filtered_data.groupby('CustomerID')['Revenue'].sum()
        fig = px.box(customer_revenue, title='Customer Revenue Distribution')
        fig.update_layout(yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Products by quantity sold
        quantity_sold = filtered_data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(quantity_sold, title='Top 10 Products by Quantity Sold')
        fig.update_layout(xaxis_title="Product", yaxis_title="Quantity Sold")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average price by product
        avg_price = filtered_data.groupby('Description')['UnitPrice'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(avg_price, title='Top 10 Products by Average Price')
        fig.update_layout(xaxis_title="Product", yaxis_title="Average Price (£)")
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by hour of day
        filtered_data['Hour'] = filtered_data['InvoiceDate'].dt.hour
        hourly_sales = filtered_data.groupby('Hour')['Revenue'].sum()
        fig = px.line(hourly_sales, title='Sales Distribution by Hour of Day')
        fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales by day of week
        filtered_data['DayOfWeek'] = filtered_data['InvoiceDate'].dt.day_name()
        daily_sales = filtered_data.groupby('DayOfWeek')['Revenue'].sum()
        # Reorder days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sales = daily_sales.reindex(days_order)
        fig = px.bar(daily_sales, title='Sales Distribution by Day of Week')
        fig.update_layout(xaxis_title="Day of Week", yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit by [Your Name]")