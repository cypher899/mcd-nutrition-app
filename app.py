import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -------------------------------
# 1. Page Setup
# -------------------------------
st.set_page_config(page_title="McDonald's India Nutrition Analyzer", layout="wide")
st.title("🍔 McDonald's India Nutrition Analyzer")
st.markdown("""
This app helps you analyze nutritional data of menu items from McDonald's India.
Use the sidebar to filter and explore insightful charts and statistics.
""")

# -------------------------------
# 2. Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\SHIVAM\.vscode\India_Menu.csv")  # Replace with actual dataset
    # For now, let's simulate a McDonald's dataset:
    df = pd.read_csv(r"C:\Users\SHIVAM\.vscode\India_Menu.csv")  # Replace with actual McD data later
    return df

df = load_data()

# Simulated McDonald's Dataset for demo
np.random.seed(42)
df = pd.DataFrame({
    'Item': [f'Item {i}' for i in range(1, 142)],
    'Category': np.random.choice(['Burger', 'Wrap', 'Beverage', 'Dessert', 'Sides'], 141),
    'Calories': np.random.randint(100, 800, 141),
    'Protein': np.random.uniform(2, 30, 141).round(1),
    'Total Fat': np.random.uniform(5, 40, 141).round(1),
    'Carbohydrates': np.random.uniform(10, 90, 141).round(1),
    'Sugar': np.random.uniform(0, 50, 141).round(1),
    'Price (INR)': np.random.randint(50, 300, 141)
})

# -------------------------------
# 3. Sidebar Filters
# -------------------------------
st.sidebar.header("🔍 Filter Menu")
selected_categories = st.sidebar.multiselect("Select Categories", options=df['Category'].unique(), default=df['Category'].unique())
cal_min, cal_max = st.sidebar.slider("Select Calorie Range", int(df['Calories'].min()), int(df['Calories'].max()), (100, 700))

filtered_df = df[
    (df['Category'].isin(selected_categories)) &
    (df['Calories'] >= cal_min) &
    (df['Calories'] <= cal_max)
]

# -------------------------------
# 4. Summary Stats
# -------------------------------
st.subheader("📊 Nutritional Overview")
st.markdown(f"Showing **{len(filtered_df)} items** after filtering.")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# Stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Calories", f"{filtered_df['Calories'].mean():.1f} kcal")
col2.metric("Average Protein", f"{filtered_df['Protein'].mean():.1f} g")
col3.metric("Average Sugar", f"{filtered_df['Sugar'].mean():.1f} g")
col4.metric("Avg. Calorie per ₹", f"{(filtered_df['Calories']/filtered_df['Price (INR)']).mean():.2f} kcal/₹")

# -------------------------------
# 5. Visual Insights
# -------------------------------
st.subheader("📈 Visual Nutrition Insights")

# Calories Distribution
st.markdown("### Calories Distribution")
fig1 = px.histogram(filtered_df, x="Calories", color="Category", nbins=30)
st.plotly_chart(fig1, use_container_width=True)

# Protein vs Calories
st.markdown("### Protein vs Calories")
fig2 = px.scatter(filtered_df, x="Calories", y="Protein", color="Category", hover_data=['Item'])
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# 6. Calorie Per Rupee Analysis
# -------------------------------
filtered_df['CaloriePerRupee'] = (filtered_df['Calories'] / filtered_df['Price (INR)']).round(2)
st.subheader("💰 Calorie per Rupee Value")
fig3 = px.bar(filtered_df.sort_values(by='CaloriePerRupee', ascending=False).head(10),
              x='Item', y='CaloriePerRupee', color='Category', title="Top 10 Items by Calorie per ₹")
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# 7. Recommended Items
# -------------------------------
st.subheader("🍟 Recommended Items")
st.markdown("Recommended based on high protein, low sugar and good calorie/₹ ratio")
recommended = filtered_df[(filtered_df['Protein'] > 15) & (filtered_df['Sugar'] < 10)]
recommended = recommended.sort_values(by='CaloriePerRupee', ascending=False).head(10)
st.dataframe(recommended[['Item', 'Category', 'Calories', 'Protein', 'Sugar', 'CaloriePerRupee']], use_container_width=True)

# -------------------------------
# 8. Downloadable Data
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.download_button("📥 Download Filtered CSV", data=filtered_df.to_csv(index=False).encode(),
                          file_name='filtered_mcd_items.csv', mime='text/csv')

# -------------------------------
# 9. Show Raw Data Toggle
# -------------------------------
with st.expander("🔎 Show Raw Full Dataset"):
    st.dataframe(df, use_container_width=True)

# -------------------------------
# 10. Footer
# -------------------------------
st.markdown("""
---
Made with ❤️ by **Your Name**  
VT Project - 2nd Year AIML
""")