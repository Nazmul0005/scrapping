import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
import pandas as pd
import datetime
import re
import json
from pathlib import Path

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PR Supermarket Price Comparison",
    page_icon="🛒",
    layout="wide"
)

# API key from environment or user input
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

# Initialize model if API key provided
model = None
if api_key:
    model = ChatOpenAI(
        model="deepseek/deepseek-r1:free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

# Create data directory if it doesn't exist
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

# List of Puerto Rico supermarkets
supermarkets = {
    "Pueblo": "https://www.daraz.com.bd/#?",
    "SuperMax": "https://www.shwapno.com/",
    "Selectos": "https://selectos.com/shopper/",
    # Add more supermarkets as needed
}

# Main title
st.title("🛒 Puerto Rico Supermarket Price Comparison")

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Scrape Circulars", "View Products", "Compare Prices", "Weekly Report"])

with tab1:
    st.header("Scrape Weekly Circulars")
    
    # Supermarket selection
    selected_markets = st.multiselect(
        "Select supermarkets to scrape",
        options=list(supermarkets.keys()),
        default=list(supermarkets.keys())[:1]
    )
    
    # Date input for circular week
    circular_date = st.date_input(
        "Circular week (usually starts on Wednesday)",
        datetime.date.today()
    )
    
    # Scrape button
    if st.button("Scrape Selected Circulars"):
        if not selected_markets:
            st.error("Please select at least one supermarket")
        else:
            all_products = []
            
            for market in selected_markets:
                try:
                    # Show scraping progress
                    st.subheader(f"Scraping {market}...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Get the URL for the selected supermarket
                    base_url = supermarkets[market]
                    status_text.text(f"Fetching circular from {base_url}...")
                    
                    # Fetch the webpage
                    response = requests.get(base_url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    })
                    response.raise_for_status()
                    
                    progress_bar.progress(30)
                    status_text.text("Parsing circular content...")
                    
                    # Parse the HTML content
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract product information 
                    # This is a simplified approach - actual implementation will need to be customized for each supermarket's HTML structure
                    progress_bar.progress(50)
                    status_text.text("Extracting product information...")
                    
                    # Process with LLM to extract product info if model is available
                    if model:
                        status_text.text("Using AI to extract product details...")
                        from langchain.schema import HumanMessage
                        
                        # Get visible text content
                        for script in soup(["script", "style"]):
                            script.extract()
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        content = '\n'.join(line for line in lines if line)
                        
                        prompt = f"""
                        Extract product information from this supermarket circular content.
                        For each product found, identify:
                        1. Product name (in Spanish)
                        2. Price
                        3. Brand (if available)
                        4. Sale end date (if available)
                        5. Category (e.g., meats, dairy, vegetables)
                        
                        Content from {market} circular:
                        {content[:5000]}
                        
                        Format the output as a JSON array of product objects.
                        Example: 
                        [
                            {{
                                "name": "Arroz Grano Mediano",
                                "price": 1.99,
                                "brand": "Rico",
                                "end_date": "2023-07-20",
                                "category": "grains"
                            }},
                            ...
                        ]
                        """
                        
                        # Get response from the model
                        messages = [HumanMessage(content=prompt)]
                        llm_response = model.invoke(messages).content
                        
                        # Try to parse JSON from the response
                        try:
                            # Extract JSON array if embedded in text
                            json_match = re.search(r'\[\s*\{.*\}\s*\]', llm_response, re.DOTALL)
                            if json_match:
                                products_json = json_match.group(0)
                            else:
                                products_json = llm_response
                                
                            products = json.loads(products_json)
                            
                            # Add supermarket name to each product
                            for product in products:
                                product['supermarket'] = market
                                product['date_scraped'] = circular_date.isoformat()
                            
                            all_products.extend(products)
                            
                            status_text.text(f"Found {len(products)} products from {market}")
                        except json.JSONDecodeError as e:
                            st.error(f"Error parsing product data: {str(e)}")
                            st.text(llm_response)
                    
                    progress_bar.progress(100)
                    
                except Exception as e:
                    st.error(f"Error scraping {market}: {str(e)}")
                    st.exception(e)
            
            if all_products:
                # Save products to CSV
                df = pd.DataFrame(all_products)
                filename = f"products_{circular_date.isoformat()}.csv"
                filepath = data_dir / filename
                df.to_csv(filepath, index=False)
                
                # Also save as JSON for easier processing
                json_filename = f"products_{circular_date.isoformat()}.json"
                json_filepath = data_dir / json_filename
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(all_products, f, ensure_ascii=False, indent=4)
                
                st.success(f"Successfully scraped {len(all_products)} products from {len(selected_markets)} supermarkets!")
                st.success(f"Data saved to {filepath} and {json_filepath}")
                
                # Display preview
                st.subheader("Preview of scraped products")
                st.dataframe(df)

with tab2:
    st.header("View Products by Supermarket")
    
    # Find all product files
    product_files = list(data_dir.glob("products_*.csv"))
    
    if not product_files:
        st.info("No product data available. Please scrape supermarket circulars first.")
    else:
        # Let user select a file
        file_options = {file.stem: file for file in product_files}
        selected_file = st.selectbox(
            "Select product data to view",
            options=list(file_options.keys())
        )
        
        if selected_file:
            # Load the selected file
            df = pd.read_csv(file_options[selected_file])
            
            # Filter options
            supermarket_filter = st.multiselect(
                "Filter by supermarket",
                options=df['supermarket'].unique(),
                default=df['supermarket'].unique()
            )
            
            if 'category' in df.columns:
                category_filter = st.multiselect(
                    "Filter by category",
                    options=df['category'].unique(),
                    default=[]
                )
            else:
                category_filter = []
            
            # Apply filters
            filtered_df = df[df['supermarket'].isin(supermarket_filter)]
            if category_filter and 'category' in df.columns:
                filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
                
            # Display products
            st.subheader(f"Products from {', '.join(supermarket_filter)}")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "Download filtered data as CSV",
                csv,
                f"filtered_{selected_file}.csv",
                "text/csv",
                key='download-csv'
            )

with tab3:
    st.header("Compare Product Prices")
    
    product_files = list(data_dir.glob("products_*.csv"))
    
    if not product_files:
        st.info("No product data available. Please scrape supermarket circulars first.")
    else:
        # Let user select a file
        file_options = {file.stem: file for file in product_files}
        selected_file = st.selectbox(
            "Select product data to compare",
            options=list(file_options.keys()),
            key="compare_file_select"
        )
        
        if selected_file:
            # Load the selected file
            df = pd.read_csv(file_options[selected_file])
            
            # Search functionality
            search_term = st.text_input("Search for products (e.g., 'arroz', 'leche', 'carne')")
            
            if search_term:
                # Use model for semantic search if available
                if model:
                    st.info("Using AI to find similar products across supermarkets...")
                    
                    from langchain.schema import HumanMessage
                    
                    # Convert products to string for the prompt
                    products_str = df.to_string(max_rows=100)
                    
                    prompt = f"""
                    I have product data from different supermarkets in Puerto Rico.
                    Find all products that match or are similar to: "{search_term}"
                    
                    Consider variations in product names, brands, and descriptions. 
                    For example, "arroz" might be listed as "arroz grano mediano", "arroz Rico", etc.
                    
                    Return the row numbers from this product table that match:
                    
                    {products_str}
                    
                    Just return a list of row numbers (index values) without any explanation.
                    """
                    
                    # Get response from the model
                    messages = [HumanMessage(content=prompt)]
                    llm_response = model.invoke(messages).content
                    
                    # Try to extract row numbers
                    try:
                        row_nums = re.findall(r'\d+', llm_response)
                        row_indices = [int(num) for num in row_nums if int(num) < len(df)]
                        
                        if row_indices:
                            filtered_df = df.iloc[row_indices]
                        else:
                            # Fallback to simple text search
                            mask = df.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)
                            filtered_df = df[mask]
                    except:
                        # Fallback to simple text search
                        mask = df.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)
                        filtered_df = df[mask]
                else:
                    # Simple text search
                    mask = df.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)
                    filtered_df = df[mask]
                
                # Display results
                if not filtered_df.empty:
                    # Group by product name (or similar products)
                    if model:
                        # Use AI to group similar products
                        st.info("Grouping similar products for comparison...")
                        
                        # Simplified grouping for the example
                        # In a real implementation, this would use more sophisticated NLP
                        grouped_df = filtered_df.sort_values(['name', 'price'])
                        
                        # Display grouped products
                        st.subheader(f"Price comparison for '{search_term}'")
                        st.dataframe(
                            grouped_df[['name', 'price', 'brand', 'supermarket']],
                            use_container_width=True
                        )
                        
                        # Find the best deals
                        if 'category' in grouped_df.columns:
                            categories = grouped_df['category'].unique()
                            for category in categories:
                                cat_df = grouped_df[grouped_df['category'] == category]
                                best_deal = cat_df.loc[cat_df['price'].idxmin()]
                                st.success(f"Best deal for {category}: {best_deal['name']} at {best_deal['supermarket']} for ${best_deal['price']}")
                    else:
                        # Simple grouping by exact name
                        st.dataframe(
                            filtered_df[['name', 'price', 'brand', 'supermarket']].sort_values('price'),
                            use_container_width=True
                        )
                else:
                    st.warning(f"No products found matching '{search_term}'")

with tab4:
    st.header("Generate Weekly Report")
    
    product_files = list(data_dir.glob("products_*.csv"))
    
    if not product_files:
        st.info("No product data available. Please scrape supermarket circulars first.")
    else:
        # Let user select a file
        file_options = {file.stem: file for file in product_files}
        selected_file = st.selectbox(
            "Select product data for report",
            options=list(file_options.keys()),
            key="report_file_select"
        )
        
        if selected_file:
            # Load the selected file
            df = pd.read_csv(file_options[selected_file])
            
            # Report generation button
            if st.button("Generate Weekly Savings Report"):
                if model:
                    st.info("Generating comprehensive analysis with AI...")
                    
                    from langchain.schema import HumanMessage
                    
                    # Prepare summary statistics
                    supermarket_counts = df['supermarket'].value_counts()
                    total_products = len(df)
                    
                    # Basic category analysis
                    if 'category' in df.columns:
                        category_stats = df.groupby(['category', 'supermarket'])['price'].agg(['mean', 'min', 'max', 'count'])
                        
                        # Find best deals per category
                        best_deals = df.loc[df.groupby('category')['price'].idxmin()]
                        
                        summary_data = {
                            "total_products": total_products,
                            "supermarket_counts": supermarket_counts.to_dict(),
                            "categories": df['category'].unique().tolist(),
                            "best_deals": best_deals[['category', 'name', 'price', 'supermarket']].to_dict(orient='records')
                        }
                    else:
                        summary_data = {
                            "total_products": total_products,
                            "supermarket_counts": supermarket_counts.to_dict()
                        }
                    
                    prompt = f"""
                    Generate a comprehensive weekly savings report for Puerto Rico supermarket price comparison.
                    
                    Summary data:
                    {json.dumps(summary_data, indent=2)}
                    
                    Please provide:
                    1. An executive summary of the findings
                    2. Best deals by category
                    3. Supermarket comparison (which has better prices overall)
                    4. Shopping recommendations for budget-conscious consumers
                    
                    Format the report with markdown headings and bullet points for clarity.
                    """
                    
                    # Get response from the model
                    messages = [HumanMessage(content=prompt)]
                    llm_response = model.invoke(messages).content
                    
                    # Display the report
                    st.markdown(llm_response)
                    
                    # Save report option
                    if st.button("Save This Report"):
                        report_filename = f"report_{selected_file.split('_')[1]}.md"
                        report_filepath = data_dir / report_filename
                        
                        with open(report_filepath, 'w', encoding='utf-8') as f:
                            f.write(f"# Weekly Supermarket Price Comparison Report\n")
                            f.write(f"**Generated on:** {datetime.date.today().isoformat()}\n\n")
                            f.write(llm_response)
                            
                        st.success(f"Report saved to {report_filepath}")
                else:
                    # Basic report without AI
                    st.subheader("Weekly Price Comparison Summary")
                    
                    # Count products by supermarket
                    st.write("Products per supermarket:")
                    st.bar_chart(df['supermarket'].value_counts())
                    
                    if 'category' in df.columns:
                        # Average prices by category and supermarket
                        pivot = pd.pivot_table(
                            df, 
                            values='price', 
                            index='category',
                            columns='supermarket',
                            aggfunc='mean'
                        )
                        
                        st.write("Average prices by category:")
                        st.dataframe(pivot)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download full data as CSV",
                        csv,
                        f"full_{selected_file}.csv",
                        "text/csv",
                        key='download-report-csv'
                    )

# Sidebar content
with st.sidebar:
    st.subheader("About")
    st.write("""
    This tool helps Puerto Rico consumers save money by comparing prices 
    across different supermarkets' weekly circulars.
    """)
    
    st.subheader("Instructions")
    st.write("""
    1. Go to 'Scrape Circulars' tab to gather latest prices
    2. View all products in the 'View Products' tab
    3. Compare prices for specific items in the 'Compare Prices' tab
    4. Generate a savings report in the 'Weekly Report' tab
    """)
    
    # Show API key status
    if model:
        st.success("✅ AI model connected (enhances product matching)")
    else:
        st.warning("⚠️ No API key provided (limited functionality)")
    
    # Add contact info
    st.subheader("Contact")
    st.info("For feedback or support, please contact the developer.")
