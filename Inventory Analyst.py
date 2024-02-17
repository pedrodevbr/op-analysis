
## This program manage the mro inventory of a company
# The goal is analyse the planned orders and decide if it will be converted in a purchase order or not

## Some assumptions are:
# Cost of process is high (Idealy I want to reduce the number of purchase orders)
# Cost of storage is low  (I have a lot of space)
# Disponibility to client of 92%
# The lead time varies for every material
# The lead time is the time between the request and the delivery 

## Data on the sheet:
# Date of last purchase order
# Data of last consumption
# Criticity
# Lead time
# Cost of material

## Calculate
# Replenishment point
# Maximum stock

## The program will be used for the following steps:
# 1 - Upload the material data from SAP
# 2 - classify the itens based on consumption 
# In the OP there are the last purchase order, criticity, lead time, virtual stock, stock, cost of material 
# calculate the optimal replenishment quantity and the maximum stock based on the best management literature
# Prz.entrg.prev. = Lead time
# 1 LTD = Last month consumption


#######################
# Import libraries
import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import plotly.express as px
import altair as alt

#######################
# Page configuration
st.set_page_config(
    page_title="Data Analysis - Inventory Management",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# Load data
@st.cache_data
def load_data():
        
    DIR_PATH = './data/'
    # op, oc, consumo, req, san, reserva
    op_df = pd.read_excel(DIR_PATH+'op.XLSX')
    oc_df = pd.read_excel(DIR_PATH+'oc.XLSX')
    consumo_df = pd.read_excel(DIR_PATH+'130.XLSX',thousands='.',decimal=',')
    #req_df = pd.read_excel(DIR_PATH+'req.XLSX')
    san_df = pd.read_excel(DIR_PATH+'san.XLSX')
    reserva_df = pd.read_excel(DIR_PATH+'reserva.XLSX')
    textos_df = pd.read_excel(DIR_PATH+'textos.XLSX')
    return op_df, consumo_df, san_df, reserva_df,oc_df,textos_df
op_df, consumo_df, san_df, reserva_df,oc_df, textos_df= load_data()

def clean_op(df):
    # turn Material into string
    df['Material'] = df['Material'].astype(str)
    # fill Criticidade with 0
    df['Criticidade'] = df['Criticidade'].fillna(0)
    # fill NºPF  with sem referencia   
    df['NºPF'] = df['NºPF'].fillna('Sem referencia')

    return df
op_df = clean_op(op_df)

def nlarge_2(row):
    # get the second largest value in the row
    for i in range(len(row)):
        if row[i] == row.max():
            row[i] = 0
    return row.max()

def clean_130(df):
    # turn Material into string
    df['Material'] = df['Material'].astype(str)
    # clean spaces
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # convert into numeric
    df.iloc[1:, 3:] = df.iloc[1:, 3:].apply(pd.to_numeric, errors='coerce')

    # create a new column with the second highest value in the row first 12 columns
    df['2nd highest'] = df.iloc[1:, 3:15].apply(lambda x: nlarge_2(x), axis=1)

    # average the last 3 values in the row
    df['Average'] = df.iloc[1:, 3:5].mean(axis=1)

    return df
consumo_df = clean_130(consumo_df)

def clean_san(df):
    san_df = pd.DataFrame()
    # turn Material into string
    san_df['Material'] = df['Material'].astype(str)
    san_df['Qty in LMR'] = df['Quantidade Material LMR']
    return san_df
san_df = clean_san(san_df)

def clean_reserva(df):
    reserva_df = pd.DataFrame()
    #filter by Com registro final == x
    df = df[df['Com registro final'] == 'X']
    # filter Data base,Descrição do Equipamento,Motivo da Reserva
    df['Material'] = df['Material'].astype(str)
    # turn data base in datetime format
    df['Last reserve'] = pd.to_datetime(df['Data base'], format='%d-%m-%Y %H:%M:%S').dt.strftime('%d-%m-%Y')
    #filter the last reserve for each material
    df = df.sort_values('Last reserve').drop_duplicates('Material', keep='last')
    reserva_df = df[['Material','Last reserve','Descrição do Equipamento','Motivo da Reserva']]

    return reserva_df
reserva_df = clean_reserva(reserva_df)

def clean_oc(df):
    oc_df = pd.DataFrame()
    # turn Material into string
    df['Material'] = df['Material'].astype(str)
    # drop cells with 'cancelado' in the column 'Texto breve'
    df = df[~df['Texto breve'].str.contains('cancelado', case=False, na=False)]
    # filter the last purchase order for each material
    df = df.sort_values('Data do documento').drop_duplicates('Material', keep='last')
    df['Last OC'] = pd.to_datetime(df['Data do documento'], format='%d-%m-%Y %H:%M:%S').dt.strftime('%d-%m-%Y')
    oc_df = df[['Material','Last OC']]


    return oc_df
oc_df = clean_oc(oc_df)

def clean_textos(df):
    # turn Material into string
    df['Material'] = df['Material'].astype(str)
    df = df.dropna(subset=['Texto OBS - pt'])
    df['Texto OBS - pt'] = df['Texto OBS - pt'].astype(str)
    #join all Texto OBS - pt for each material
    obs_df = df.groupby('Material')['Texto OBS - pt'].apply(lambda x: '\n'.join(x))
    
    return obs_df

textos_df = clean_textos(textos_df)
textos_df.head()

def days_in_op(df):
    # convert AbertPl to datetime
    df['AbertPl'] = pd.to_datetime(df['AbertPl'])
    # calculate the days
    df['AbertPl'] = dt.datetime.now() - df['AbertPl']
    df['AbertPl'] = df['AbertPl'].dt.days
    return df['AbertPl']

def summary(op_df, san_df, consumo_df, reserva_df):
    
    #OP
    df = pd.DataFrame()
    df["Action"] = '-'
    df["Material"] = op_df["Material"]
    df["Description"] = op_df["Nº do material"]
    
    df["Policy"] = op_df['TpM']
    df['PR'] = op_df['Pt.reabast.'].astype(int)
    df['Max'] = op_df['Est.máximo'].astype(int)
    #df["Policy"] = df['TpM'].astype(str) + ' = ' + df['Pt.reabast.'].astype(str) + '/' + df['Est.máximo'].astype(str)
    df["GM"] = op_df["GrpMercads."]
    df['Status'] = op_df['GMRP']
    df['Criticity'] = op_df['Criticidade']
    df['Reserved'] = op_df['Quantidade Tp 3']
    df['Order Value'] = op_df['Valor total da Orden Planejada'].round().astype(int)
    df['Virtual Stock'] = op_df['Saldo Virtual'].astype(float)

    df['Qty OP'] = op_df['Qtd.ordem'].astype(float)
    df["Days in planned order"] = days_in_op(op_df)
    df['Reference'] = op_df['NºPF']

    #OC
    # merge last purchase order with the df
    df = df.merge(oc_df, on='Material', how='left')
    #SAN
    df = df.merge(san_df, on='Material', how='left')
    #RESERVA
    # merge last reserve with the df
    df = df.merge(reserva_df, on='Material', how='left')
    #CONSUMO
    df = df.merge(consumo_df[['Material','2nd highest']], on='Material', how='left')
    #TEXTOS
    df = df.merge(textos_df, on='Material', how='left')

    return df

data_df = summary(op_df, san_df, consumo_df,reserva_df)

def define_action(df):

    if df['Qty in LMR'] > 0 and (df['Status'] == 'FRAC' or df['Status'] == 'ANA'):
        df['Action'] = 'Abrir consulta SMIT'
    elif df["Status"]=='AD':
        df['Action'] = 'Solicitar cotação ou verificar se o fornecedor esta cadastrado'
    elif df['Policy']=='ND':
        df['Action'] = 'Excluir OP, sem política'
    elif df['2nd highest'] > df['PR'] and df['Status']=='ZSTK' and df['Policy']=='ZM':
        df['Action'] = f'Aumentar PR para {df["2nd highest"]} e ajustar Est. Máximo'
    elif df['2nd highest'] < 0.8*df['PR'] and df['Status']=='ZSTK' and df['Policy']=='ZM':
        df['Action'] = f'Diminuir PR para {df["2nd highest"]} e ajustar Est. Máximo'
    elif df['Status'] == 'ZSTK' and df['Reference']=='Sem referencia':
        df['Action'] = 'Atribuir referencia'
    if "Sustentabilidade" in str(df['Texto OBS - pt']):
        df['Action'] = 'Incluir clausula sustentabilidade no pedido de compra'
    return df

data_df = data_df.apply(define_action, axis=1)

# General info
# Average AbertPl time
# make today - AbertPl and calculate the average
average_op_time = data_df["Days in planned order"].mean()

# Group per GMRP and TpM and count the number of each
grouped_status = data_df['Status'].value_counts().sort_values(ascending=False)
grouped_policy = data_df['Policy'].value_counts().sort_values(ascending=False)

# top 10% of the most expensive materials wita a $ in front
ten_percent = data_df['Order Value'].quantile(0.9)
most_expensive = data_df[data_df['Order Value'] > ten_percent].sort_values('Order Value', ascending=False)
most_expensive['Order Value'] = most_expensive['Order Value']

pr_1 = data_df[(data_df['PR'] == 1) & (data_df['Policy'].isin(['ZM', 'ZE']))][['Material','Description', 'Policy','Order Value']]

###### Streamlit ######

# Create a Streamlit app
st.title("OP ANALYSIS - SECTOR 31")

# Display general info
# insert a container on the left

general_info, specific_info, actions = st.tabs(['GenInfo','SpecInfo','Actions'])

with general_info:
        
    st.subheader("General Info")

    st.write(f"* Average time on planned order =  **{int(average_op_time)} days**")
    st.write("* Top 10% most expensive OP ")
    st.dataframe(most_expensive[['Material','Description', 'Order Value']],use_container_width=True,hide_index=True)
    
    #filter by Pt.reabast. = 1 and TpM ZM, ZE
    st.write("* ZM/ZE with PR=1")
    st.dataframe(pr_1,use_container_width=True,hide_index=True)

    c1,c2 = st.columns([1,1])
    c1.subheader("Materials by GMRP")
    c1.bar_chart(grouped_status)
    c2.subheader("Materials by Policy")
    c2.bar_chart(grouped_policy)

with specific_info:
    # Add a filter to the DataFrame
    st.subheader("Filter inventory data")

    selected_material = st.multiselect("Select material:", data_df["Material"].unique())
    # add a filter GMRP
    #selected_gmrp = st.multiselect("Select GMRP:", op_df["GMRP"].unique())

    if selected_material:
        with st.container():
            filtered_df = data_df[data_df["Material"].isin(selected_material)]
            # get Nº do material out of the df
            name = filtered_df['Action'].unique()
            st.write(f"Ação: **{name[0]}**")
            specs, graph = st.columns([1, 2])
            #Drop headers from transposed table
            # filtered_df = filtered_df.drop(columns=['Material','Description','Policy','GM','Status','Criticidade','Reserved','Order Value','Virtual Stock','Qty OP','Days in planned order','Reference','Last OC','Qty in LMR'])
            filtered_df['Order Value'] = filtered_df['Order Value'].apply(lambda x: f'$ {x:,.2f}')

            specs.dataframe(filtered_df[['Material','Description','Policy','PR','Max','GM','Status','Criticity','Reserved','Order Value','Virtual Stock','Qty OP','Days in planned order','Reference','Last OC','Qty in LMR']].transpose(),use_container_width=True,height=500,)
            
            # plot consumption data from selected material
            # each row is a different material and each column is a different consumption value
            data = consumo_df[consumo_df['Material'].isin(selected_material)].iloc[:, 3:-2].dropna(axis=1)
            # turn data into vector and reverse
            data_vector = pd.Series(data.values.flatten()[::-1])
            # Create a new DataFrame for Plotly
            df = pd.DataFrame({'Demand': data_vector, 'LT': data_vector.index})

            # Create a line chart with Plotly
            fig = px.line(df, x='LT', y='Demand', title='Consumption', labels={'LT': 'Lead Time', 'Demand': 'Demand'})

            # Display the Plotly chart in Streamlit
            graph.plotly_chart(fig)

            graph.write(filtered_df['Texto OBS - pt'].values[0])
            # plot the last reserve: Last reserve, Descrição do Equipamento, Motivo da Reserva
            st.write("Ultima reserva")
            st.dataframe(filtered_df[['Last reserve', 'Descrição do Equipamento', 'Motivo da Reserva']],use_container_width=True,hide_index=True)

with actions:
    st.subheader("Actions")
    st.write("Here you can take actions on the inventory data")
    

    policy_tab, status_tab, gm_tab = st.columns([1,1,1])
    policy = policy_tab.multiselect("Select Policy:", data_df["Policy"].unique())
    if not policy:
        policy = data_df["Policy"].unique()

    status = status_tab.multiselect("Select Status:", data_df["Status"].unique())
    if not status:
        status = data_df["Status"].unique()
        
    # filter by the 4 first digits of GM
    # Convert 'GM' column to string
    data_df['GM'] = data_df['GM'].astype(str)
    
    # Now you can slice the first 4 characters
    gm = gm_tab.multiselect("Select GM:", data_df["GM"].str[:4].unique())
    if not gm:
        gm = data_df["GM"].str[:4].unique()

    # Filter the DataFrame
    data_filtered_df = data_df[(data_df['Policy'].isin(policy)) & (data_df['Status'].isin(status)) & (data_df['GM'].str[:4].isin(gm))]
    


    st.dataframe(data_filtered_df)


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

#onsumo_df = clean_table(consumo_df)

# Calculate the CV² for non-zero values each row, do not consider empty values
#consumo_df['CV²'] = (consumo_df[consumo_df!=0].iloc[:, 3:].std(axis=1) / consumo_df[consumo_df!=0].iloc[:, 3:].mean(axis=1)) ** 2
# Fill the empty values with 0 
# se há apenas 1 valor não nulo, o CV² é 0
#consumo_df['CV²'] = consumo_df['CV²'].fillna(0)

# Calculate the average time between non-zero values for each row
def avg_zero_seq(row):
    zero_lengths = []
    count = 0
    for i in range(len(row)):
        if row[i] == 0:
            count += 1
        else:
            if i > 0 and row[i-1] != 0:
                zero_lengths.append(0)
            if count > 0:
                zero_lengths.append(count)
                count = 0
    if count > 0:
        zero_lengths.append(count)
    return np.mean(zero_lengths) if zero_lengths else np.nan

# Calculate the average length of repeating zeros between non-zero values for each row
#consumo_df['Avg_Time_Between_Demands'] = consumo_df.iloc[:, 3:].apply(avg_zero_seq, axis=1)

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
#consumo_df['Classification'] = consumo_df.apply(classify, axis=1)

