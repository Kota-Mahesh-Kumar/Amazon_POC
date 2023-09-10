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

def data_for_streamlit(excel_path):
    data = pd.read_excel(excel_path, sheet_name = 0)
    data.drop("Unnamed: 0", axis = 1, inplace = True)
    data = data[1:]
    data.columns = data.iloc[0]
    data.columns.values[1:9] = ['IR_' + col   for col in data.columns[1:9]]
    #data.columns.values[9:18] = ['Amazon_Technical_' + col   for col in data.columns[9:18]]

    data = data.iloc[1:]
    data.reset_index(drop=True, inplace=True)
    modified_values = rename_the_columns(data)
    total_columns = data.columns.tolist()
    mapped_before_columns = total_columns[:18]
    data.columns = mapped_before_columns + modified_values
    return data, data.columns

def rename_the_columns(df):
    names_alterated_list = ['Title', 'Description', 'Bullet 1', 'Bullet 2', 'Bullet 3', 'Bullet 4', 'Bullet 5', 'Bullet 6', 'Bullet 7', 'Bullet 8', 
                        'Bullet 9', 'A+ Text 1', 'A+ Text 2', 'A+ Text 3', 'A+ Text 4', 'A+ Text 5', 'A+ Text 6', 'A+ Text 7', 'A+ Text 8', 'A+P 1', 'A+P 2', 'A+P 3', 'A+P 4', 'A+P 5', 'A+P 6', 'A+P 7', 'A+P 8', 'A+P 9',
                        'MI', 'AI 1', 'AI 2', 'AI 3', 'AI 4', 'AI 5', 'AI 6', 'AI 7', 'AI 8', 'AI 8-1', 'AI 8-2', 'AI 9', 'AI 10']
    second_list = df.columns
    second_list = second_list[18:]
    modified_values = []

    for i in range(0, len(names_alterated_list)):
        prefix = str(names_alterated_list[i])  + '_'
        group_values = second_list[i * 8:(i + 1) * 8]  
        
        # Add the prefix to each value in the group
        modified_group = [prefix + value for value in group_values]
        
        # Append the modified group to the result list
        modified_values.extend(modified_group)
    return modified_values


df_original, columns = data_for_streamlit(excel_path)
asin_list = df_original['ASIN'].tolist() 

critical_attributes_list = df_original.columns[9:16].tolist()

col1, col2 = st.columns(2)
with col1:
    asin = st.selectbox("ASIN",asin_list, key = "asin_number")
with col2:
    critical_attribute = st.selectbox("Critical Attribute",critical_attributes_list, key = "critical_attribute")


if 'asin_number' not in st.session_state:
    st.session_state['asin_number'] = asin_list[0]

if "critical_attribute" not in st.session_state:
    st.session_state['critical_attribute'] = critical_attributes_list[0]

asin_key = st.session_state['asin_number']
critical_attributes_key = st.session_state['critical_attribute']

def dataframe_after_filtering(df,asin_key,critical_attributes_key):
    asin_df_selected = df[df['ASIN'] == asin_key]
    selected_columns = [col for col in df.columns if critical_attributes_key in col]
    final_output = selected_columns
    dataframe_for_attributes_asin = asin_df_selected[selected_columns]
    return dataframe_for_attributes_asin, len(final_output), selected_columns

dataframe_for_attributes_asin, length, selected_columns = dataframe_after_filtering(df_original,asin_key,critical_attributes_key)
st.write('')
st.write('')
st.write('')
st.dataframe(dataframe_for_attributes_asin, use_container_width = True)
















