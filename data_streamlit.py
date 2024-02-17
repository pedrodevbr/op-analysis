

import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt


DEBUG = False

op_df = pd.read_excel('op.XLSX')

# Initializing keys
analysis_df = op_df[['Material','Nº do material']].copy()

# Days in OP (today minus the creation date)
analysis_df.loc[:,'Dias em OP'] = (dt.datetime.today() - op_df['AbertPl']).dt.days

consumo_df = pd.read_excel('201.XLSX')

# aggregate net consumption by material and by type of movement
# sum Qtd.  UM registro by tipo de movimento 201 plus z33 
def aggregate_consumption(df):
    df = df.groupby(['Material','Tipo de movimento']).agg({'Qtd.  UM registro': 'sum'}).unstack(fill_value=0)
    df.columns = df.columns.droplevel()
    return df

print(aggregate_consumption(consumo_df).head())
# 

def classify(analysis_df,consumo_df):

    # Clean table
    def clean_table(df):
        # Clean empty spaces
        df = consumo_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        # Remove dots 
        df = df.apply(lambda x: x.str.replace('.', '') if x.dtype == "object" else x)
        # Replace commas with dots
        df = df.apply(lambda x: x.str.replace(',', '.') if x.dtype == "object" else x)
        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='ignore') 
        return df

    consumo_df = clean_table(consumo_df)

    # Calculate the CV² for non-zero values each row, do not consider empty values
    consumo_df['CV²'] = (consumo_df[consumo_df!=0].iloc[:, 3:].std(axis=1) / consumo_df[consumo_df!=0].iloc[:, 3:].mean(axis=1)) ** 2
    # Fill the empty values with 0 
    # se há apenas 1 valor não nulo, o CV² é 0
    consumo_df['CV²'] = consumo_df['CV²'].fillna(0)

    # Calculate the average time between non-zero values for each row
    def avg_zero_seq(row):
        zero_lengths = []
        count = 0
        for i in range(len(row)):
            if row.iloc[i] == 0:
                count += 1
            else:
                if i > 0 and row.iloc[i-1] != 0:
                    zero_lengths.append(0)
                if count > 0:
                    zero_lengths.append(count)
                    count = 0
        if count > 0:
            zero_lengths.append(count)
        return np.mean(zero_lengths) if zero_lengths else np.nan

    # Calculate the average length of repeating zeros between non-zero values for each row
    consumo_df['Avg_Time_Between_Demands'] = consumo_df.iloc[:, 3:].apply(avg_zero_seq, axis=1)

    # Classification with these parameters:
    # CV² consumption > 0.49 and Average time between consumption > 1,32 = Esporádico
    # CV² consumption > 0.49 and Average time between consumption < 1,32 = Intermitente
    # CV² consumption < 0.49 and Average time between consumption > 1,32 = Erratico
    # CV² consumption < 0.49 and Average time between consumption < 1,32 = Suave

    # Classify each row
    def classify(row):
        #if row length < 3 then it is not possible to classify
        if len(row.dropna())-5 < 3:
            return 'Menos de 3 LTs'
        elif row['CV²'] > 0.49 and row['Avg_Time_Between_Demands'] > 1.32:
            return 'Esporádico'
        elif row['CV²'] > 0.49 and row['Avg_Time_Between_Demands'] < 1.32:
            return 'Intermitente'
        elif row['CV²'] < 0.49 and row['Avg_Time_Between_Demands'] > 1.32:
            return 'Errático'
        else:
            return 'Suave'
    consumo_df['Classification'] = consumo_df.apply(classify, axis=1)

    #join tables
    analysis_df = analysis_df.join(consumo_df['Classification'])

    return analysis_df

analysis_df = classify(analysis_df,consumo_df)


if DEBUG: 
    print(analysis_df.head())

    st.set_page_config(page_title="Ordens Planejadas",layout="wide")

    st.title('Dashboard de Ordens Planejadas')

    # Get the material code from the user
    material_code = st.text_input('Enter the material code')

    # Filter the DataFrame based on the material code
    filtered_df = op_df[op_df['Material'] == int(material_code)]

    # Check if the DataFrame is empty
    if filtered_df.empty:
        st.write('No data to display.')
    else:
    # Display the filtered DataFrame
        st.table(filtered_df)

    # Add container
    with st.container():
        # Count the number of materials in each class
        class_counts = analysis_df['Classification'].value_counts()

        # Create a bar chart
        st.bar_chart(class_counts)

    # Calculate the average of 'Dias em OP'
    average_days = analysis_df['Dias em OP'].mean()

    # Display the average in the sidebar (rount to int)
    st.sidebar.markdown(f"**Media de Dias em OP:** {average_days:.0f}")

    with st.container():
        # Display the consumption graph
        st.line_chart(consumo_df[material_code].iloc[:, 3:])
  

