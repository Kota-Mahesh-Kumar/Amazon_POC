import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
from typing import List, Tuple
import matplotlib.pyplot as plt




# Read the data
excel_path =r"Amazon_Dimension_analysis_Dashboard_v1.xlsx"
# df = pd.read_excel(excel_path, sheet_name = 0)

# st.title("Dimension Analysis")


# def set_page_config():
#     st.set_page_config(
#         page_title="Dimension Analysis",
#         page_icon="",
#         layout="wide",
#         initial_sidebar_state="expanded",
#     )
    # st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)

def data_for_streamlit(excel_path):
    data = pd.read_excel(excel_path, sheet_name = 0)
    data.drop("Unnamed: 0", axis = 1, inplace = True)
    data = data[1:]
    data.columns = data.iloc[0]
    data.columns.values[1:9] = [col + '_mf' for col in data.columns[1:9]]
    data = data.iloc[1:]
    data.reset_index(drop=True, inplace=True)

    return data, data.columns

def plots_by_area():
    col1, col2 = st.columns(2)

    data = {
        "Category": ["Electronics", "Home & Kitchen", "Grocery & Gourmet", "Beauty", "Health & Personal Care"],
        "Total Count": [10, 15, 20, 35, 21],
        "Mismatch Count": [5, 8, 12, 11, 6],
    }

    df = pd.DataFrame(data)
    fig1, ax1 = plt.subplots(figsize=(10, 6))  # Adjust the dimensions as needed

    x = np.arange(len(df))
    bar_width = 0.35

    # Create bars for Match Count
    bar1 = ax1.bar(x - bar_width/2, df["Total Count"], bar_width, label="Total Count")

    # Create bars for Mismatch Count
    bar2 = ax1.bar(x + bar_width/2, df["Mismatch Count"], bar_width, label="Mismatch Count")

    # Set the x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Category"], rotation=45,fontsize = 12 )

    # Set the y-axis label
    ax1.set_yticks([])

    ax1.set_title("Counts by Category", fontsize=16)
    ax1.legend(loc='upper left')   
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['left'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)


    # Add labels to the bars
    for bar in [bar1, bar2]:
        for rect in bar:
            height = rect.get_height()
            ax1.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    col1.pyplot(fig1)


    stacked_data = {
        "Category": ["Length", "Breadth", "Height", "Radius", "Item Weight", "Net Quantity", "Material", "Colour"],
        "Data NA": [5, 2, 0, 3, 0, 1, 0, 2],
        "Match Percentage(%)": [10, 15, 5, 8, 12, 6, 9, 7],
        "Mismatch Percentage(%)": [15, 12, 8, 5, 18, 10, 13, 11],
    }

    stacked_df = pd.DataFrame(stacked_data)

    fig2, ax2 = plt.subplots(figsize=(10, 6))  # Adjust the dimensions as needed

    bottom = np.zeros(len(stacked_df))

    for col_name in ["Data NA", "Mismatch Percentage(%)", "Match Percentage(%)"]:
        ax2.bar(stacked_df["Category"], stacked_df[col_name], label=col_name, bottom=bottom)
        bottom += stacked_df[col_name]

    # ax2.set_ylabel("Percentage (%)")  

    total_values = stacked_df[["Data NA", "Mismatch Percentage(%)", "Match Percentage(%)"]].sum(axis=1)

    percentages = (total_values / total_values.sum()) * 100

    # Set y-axis ticks and labels from 0 to 100
    ax2.set_yticks(np.arange(0, 101, 10))
    ax2.set_yticklabels([f"{int(percentage)}%" for percentage in np.arange(0, 101, 10)])

    ax2.set_xticklabels(stacked_df["Category"], rotation=45, ha="right")
    ax2.set_title("Counts by Critical Attributes", fontsize=16)
    ax2.legend()
    col2.pyplot(fig2)

st.set_page_config(layout="wide")


st.markdown(
    """
    <h1 style="text-align:center;">Critical Attribute Analysis</h1>
    """,
    unsafe_allow_html=True
)

# set_page_config()
df_original, columns = data_for_streamlit(excel_path)

asin_list = df_original['ASIN'].tolist() 
asin_list.insert(0, "All")

critical_attributes_list = df_original.columns[9:16].tolist()
critical_attributes_list.insert(0, 'All')

col1, col2, col3, col4 = st.columns(4)
with col1:
    cat = st.selectbox("Choose Category", ["All","Electronics", "Home & Kitchen", "Grocery & Gourmet", "Beauty", "Health & Personal Care"])
with col2:
    sub_cat = st.selectbox("Choose Sub-Category", ["All", 'Sofas & Couches', 'Tv & Entertainment Units', 'Jars & Containers', 'Drinks', 'Container Insulators', 'Tea lights'])
with col3:
    asin = st.selectbox("ASIN",asin_list)
with col4:
    critical_attribute = st.selectbox("Critical Attribute",critical_attributes_list )

st.write('')
st.write('')

def calculate_kpis(data: pd.DataFrame) -> List[float]:
    total_asins = len(data["ASIN"])
    match_count = "16(57%)"
    mismatch_count = "12 (43%)"
    data_na_inspection = "22"
    data_na_amazon = "12"
    return [total_asins, match_count, mismatch_count, data_na_inspection, data_na_amazon]

def display_kpi_metrics(kpis: List[float], kpi_names: List[str]):
    for i, (col, (kpi_name, kpi_value)) in enumerate(zip(st.columns(5), zip(kpi_names, kpis))):
        col.metric(label=kpi_name, value=kpi_value)

kpi_names = ["Total No. of CAs", "No. of Match", "No. of Mismatch", "Data NA (Inspection)", "Data NA (Amazon)"]
kpis = calculate_kpis(df_original)
display_kpi_metrics(kpis,kpi_names )

st.write('')
st.write('')

plots_by_area()

st.write('')

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        flex-direction: column;
    }
    .block-container {
        display: flex;
        flex-direction: column-reverse;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
    .centered-header {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
st.markdown('<h1 class="centered-header">Data Table</h1>', unsafe_allow_html=True)

dataframe_to_display = df_original.iloc[:,0:18]
dataframe_to_display.rename(columns={'ASIN': 'ASIN', 'Length_mf': 'IR_Length', 'Breadth_mf' :'IR_Breadth', 'Height_mf' : 'IR_Height', 'Radius_mf' :'IR_Radius', 'Item Weight_mf': 'IR_Item Weight', 'Net Quantity_mf': 'IR_Net Quantity', 'Material_mf' : 'IR_Material','Colour_mf' :'IR_Colour'}, inplace=True)
def extract_numerical(s):
    try:
        s = "".join(c for c in s if c.isdigit() or c == ".")
        return float(s) if s else None  # Convert to float if not empty, else return None
    except (TypeError, ValueError):
        return None

def create_comparison_dataframe(df):
    columns_to_compare = ["Length", "Breadth", "Height", "Colour", "Material", "Radius", "Item Weight", "Net Quantity" ]
    comparison_data = {"ASIN": df["ASIN"]}
    
    for column in columns_to_compare:
        comparison_data[column] = [
            "NONE" if df[column].iloc[i] is None or df[f"IR_{column}"].iloc[i] is None
            else "MATCH" if extract_numerical(df[column].iloc[i]) == extract_numerical(df[f"IR_{column}"].iloc[i])
            else "MISMATCH"
            for i in range(len(df))
        ]

    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df
comparison_df = create_comparison_dataframe(dataframe_to_display)
styled_comparison_df = comparison_df.style.applymap(lambda cell: 'background-color: green' if cell == 'MATCH' else ('background-color: pink red' if cell == 'MISMATCH' else ''), subset=comparison_df.columns[1:])
st.dataframe(styled_comparison_df, use_container_width = True)

# # Display the styled DataFrame
# try:
#    st.dataframe(dataframe_to_display)
# except:
#     st.write(dataframe_to_display)


def download_dataframe():
    csv = dataframe_to_display.to_csv(index=False)
    return csv

csv = download_dataframe()
st.download_button(
    label="Download Data as CSV",
    data=csv,
    key="download_dataframe",
    file_name="my_dataframe.csv",
    mime="text/csv"
)

# def highlight_missing_data(s):
#     """Highlight missing data in red."""
#     missing = s.isna()
#     return ['background-color: grey' if m else '' for m in missing]
# # styled_df = dataframe_to_display.style.apply(highlight_missing_data)

# def make_column_bold(data, column_name):
#     """
#     Make a specific column bold using CSS.
#     """
#     styles = []
#     for idx, value in enumerate(data[column_name]):
#         style = f'font-weight: bold;' if idx == 0 else ''  # Make the first row bold
#         styles.append(style)
#     return styles

# # styled_df = dataframe_to_display.style.apply(lambda x: ['font-weight: bold' if x.name == "ASIN" else '' for _ in x], axis=0)

# styled_df = dataframe_to_display.style.applymap(lambda x: 'font-weight: bold')
