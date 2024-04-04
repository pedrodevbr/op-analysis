################################
# Analise de ordens planejadas #
################################


#######################
# Import libraries
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import altair as alt
#import matplotlib.pyplot as plt
import streamlit as st

# import image
from IPython.display import Image as IPythonImage



#######################
# CONSTANTS

DEBUG = True
#######################
# data da extração de dados SAP

data_OP = dt.datetime.now().strftime('%Y-%m')
data_OP = '2024-03'
DIR_PATH = f'./data/{data_OP}/'

# create a new directory
import os
if not os.path.exists(DIR_PATH):
    os.makedirs(DIR_PATH)

print(f'Extrair dados de {data_OP}...')

#######################
# Page configuration
st.set_page_config(
    page_title="Data Analysis - Inventory Management",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded")

### Definir quais colunas serão utilizadas de cada dataframe ###

## Unico material por linha
# OP        - AbertPl,Material,Nº do material,PlMRP,GrpMercads.,GMRP,PEP,Criticidade,TpM,Demanda,Saldo Virtual,Pt.reabast.,Est.máximo,Qtd.ordem,CMM,Estq.segurança,Quantidade Tp 1,Quantidade Tp 3,Quantidade Tp 6,Valor total da Orden Planejada,NºPF
# SAN       - Material, Stat.mat.todos cent.,Setor de atividade,LMR?,Quantidade Material LMR
# 130       - Material, Prz.entrg.prev., 1 LTD,2 LTD,3 LTD,4 LTD,5 LTD,6 LTD,7 LTD,8 LTD,9 LTD,10 LTD,11 LTD,12 LTD,13 LTD,14 LTD,15 LTD,16 LTD,17 LTD,18 LTD,19 LTD,20 LTD,21 LTD,22 LTD,23 LTD,24 LTD,25 LTDutf-8
# 0053      - Material, Peso bruto, Volume

## Mais de um material por linha (join)
# OC        - Material, Texto breve, Documento de compras, data do documento,Fornecedor/centro fornecedor
# 0028      - Material, Tipo de reserva, Centro custo,data base,Nome do usuário,Cód. Localização,Descrição do Equipamento,Material,Texto,Com registro final,Item foi eliminado,Motivo da Reserva
# textos    - Material, Texto OBS - pt,Texto DB - pt

## Verificar se os arquivos existem
for file in ['OP.XLSX', 'SAN.XLSX', '130.XLSX', '0053.XLSX', 'OC.XLSX', '0028.XLSX', 'textos.XLSX']:
    if not os.path.exists(f'{DIR_PATH}{file}'):
        print(f'Arquivo {file} não encontrado')

## Carregar dados com as colunas selecionadas de arquivos em excel
@st.cache_data
def load_data():
    
    op      = pd.read_excel(f'{DIR_PATH}OP.XLSX',       usecols=['AbertPl','Material','Nº do material','PlMRP','GrpMercads.','GMRP','PEP','Criticidade','TpM','Demanda','Saldo Virtual','Pt.reabast.','Est.máximo','Qtd.ordem','CMM','Estq.segurança','Quantidade Tp 1','Quantidade Tp 3','Quantidade Tp 6','Valor total da Orden Planejada','NºPF'])
    san     = pd.read_excel(f'{DIR_PATH}SAN.XLSX',      usecols=['Material','Stat.mat.todos cent.','Setor de atividade','LMR?','Quantidade Material LMR'])
    t0053   = pd.read_excel(f'{DIR_PATH}0053.XLSX',     thousands='.', decimal=',', usecols=['Material','Peso bruto','Volume'])
    oc      = pd.read_excel(f'{DIR_PATH}OC.XLSX',       usecols=['Material','Texto breve','Documento de compras','Data do documento','Fornecedor/centro fornecedor'])
    t0028   = pd.read_excel(f'{DIR_PATH}0028.XLSX',     usecols=['Material','Tipo de reserva','Centro custo','Data base','Nome do usuário','Cód. Localização','Descrição do Equipamento','Material','Texto','Com registro final','Item foi eliminado','Motivo da Reserva'])
    textos  = pd.read_excel(f'{DIR_PATH}textos.XLSX',   usecols=['Material','Texto OBS - pt','Texto DB - pt', 'Texto - pt','Texto REF LMR'])
    t130    = pd.read_excel(f'{DIR_PATH}130.XLSX',      nrows=10000,   thousands='.', decimal=',', usecols=['Material','Prz.entrg.prev.','1 LTD','2 LTD','3 LTD','4 LTD','5 LTD','6 LTD','7 LTD','8 LTD','9 LTD','10 LTD','11 LTD','12 LTD','13 LTD','14 LTD','15 LTD','16 LTD','17 LTD','18 LTD','19 LTD','20 LTD','21 LTD','22 LTD','23 LTD','24 LTD','25 LTD'])
    
    return op, san, t130, t0053, oc, t0028, textos

op, san, t130, t0053, oc, t0028, textos = load_data()

### Ajustar e limpar dataframes ###

## Converter Material para string
for df in [op, san, t130, t0053, oc, t0028, textos]:
    df['Material'] = df['Material'].astype(str)

t0053['Material'] = t0053['Material'].str.replace('.0','')

df = op
## Juntar dataframes com um material por linha
df = df.merge(san, on='Material', how='left')
df = df.merge(t130, on='Material', how='left')
df = df.merge(t0053, on='Material', how='left')

## Converter colunas para numerico
lts = ['1 LTD','2 LTD', '3 LTD', '4 LTD', '5 LTD', '6 LTD', '7 LTD', '8 LTD', '9 LTD', '10 LTD', '11 LTD', '12 LTD', '13 LTD', '14 LTD', '15 LTD', '16 LTD', '17 LTD', '18 LTD', '19 LTD', '20 LTD', '21 LTD', '22 LTD', '23 LTD', '24 LTD', '25 LTD']
if set(lts).issubset(df.columns):
    df[lts] = df[lts].apply(pd.to_numeric, errors='coerce')

outros_num = ['PEP','Criticidade','Saldo Virtual','Pt.reabast.','Est.máximo','Qtd.ordem','CMM','Estq.segurança','Quantidade Tp 1','Quantidade Tp 3','Quantidade Tp 6','Valor total da Orden Planejada','Quantidade Material LMR']
if set(outros_num).issubset(df.columns):
    df[outros_num] = df[outros_num].apply(pd.to_numeric, errors='coerce')

# Converter colunas para datetime
datas = ['AbertPl','data do documento','Data base']
if set(datas).issubset(df.columns):
    df[datas] = df[datas].apply(pd.to_datetime, errors='coerce')

# Volume e peso por OP convertidos para int e fillna(0)
df['Volume da OP']      = (df['Volume'] * df['Qtd.ordem']).fillna(0).astype(int)
df['Peso bruto da OP']  = (df['Peso bruto'] * df['Qtd.ordem']).fillna(0).astype(int)

## Dias em op
df['Dias em OP'] = (dt.datetime.now() - df['AbertPl']).dt.days

## Ação
df['ação'] = ''
# reordenar colunas
df = df[['Dias em OP', 'Material', 'Nº do material', 'GMRP', 'Criticidade', 'TpM', 'Demanda', 'GrpMercads.', 'Saldo Virtual', 'Pt.reabast.', 'Est.máximo', 'Qtd.ordem', 'NºPF', 'Estq.segurança', 'Quantidade Tp 1', 'Quantidade Tp 3', 'Quantidade Tp 6', 'ação', 'Valor total da Orden Planejada', 'Setor de atividade', 'Quantidade Material LMR', 'Prz.entrg.prev.', '1 LTD', '2 LTD', '3 LTD', '4 LTD', '5 LTD', 'Volume da OP', 'Peso bruto da OP']]

## converter para string
df['GrpMercads.'] = df['GrpMercads.'].astype(str)

#df.to_excel(f'{DIR_PATH}OP_analisado.xlsx', index=False,freeze_panes=(1,0))

setor_atividade = pd.DataFrame([27,28,29,30,31,33], columns=['Setor de atividade'])
setor_atividade['Setor de atividade'] = setor_atividade['Setor de atividade'].astype(str)

#filrar setor de atividade - 27,28,29,30,31,33
df = df[df['Setor de atividade'].isin([27,28,29,30,31,33])]

# Streamlit

st.title('Análise de Ordens Planejadas')

st.write('## Dados')

geral_tab, setor_tab, individual_tab = st.tabs(['Geral','por Setor','por Material'])

# Set filters for data
# Divide by GrpMercads., GMRP, TpM, setor de atividade
st.sidebar.markdown('## Filtros')
grp_merc = st.sidebar.multiselect('Selecione os grupos de mercadorias', df['GrpMercads.'].unique())
gmrp = st.sidebar.multiselect('Selecione os GMRP', df['GMRP'].unique())
tpm = st.sidebar.multiselect('Selecione a Política', df['TpM'].unique())
setor = st.sidebar.multiselect('Selecione o setor de atividade', df['Setor de atividade'].unique())
dias_em_op = st.sidebar.multiselect('Dias em OP < ', [30, 60, 90, 180])

# Treat each filter 
if len(grp_merc) == 0:
    grp_merc = df['GrpMercads.'].unique()
if len(gmrp) == 0:
    gmrp = df['GMRP'].unique()
if len(tpm) == 0:
    tpm = df['TpM'].unique()
if len(setor) == 0:
    setor = df['Setor de atividade'].unique()
if len(dias_em_op) == 0:
    dias_em_op = [1000]

filtered_df = df[
    (df['GrpMercads.'].isin(grp_merc)) &
    (df['GMRP'].isin(gmrp)) &
    (df['TpM'].isin(tpm)) &
    (df['Setor de atividade'].isin(setor)) &
    (df['Dias em OP'] <= dias_em_op[0]) 
]

with geral_tab:

    # Show resume of filtered data 
    # Mean of Dias em OP and sum of Valor total da Orden Planejada
    st.write('#### Resumo dos dados filtrados')
    st.write(f"Média de dias em OP: {int(filtered_df['Dias em OP'].mean())} dias")
    st.write(f"Soma do valor em OP: $ {filtered_df['Valor total da Orden Planejada'].sum():,.2f}")
    st.write(f"Quantidade de itens: {filtered_df.shape[0]}")
    st.write(f"Quantidade de itens com LMR: {filtered_df[filtered_df['Criticidade'] > 0].shape[0]}")

    # Show data
    st.write(filtered_df)

with setor_tab:

    dias_filtered = dias_em_op[0]
    filtered_df = filtered_df.fillna(0)
    # media de dias em op round to 2 decimal places
    setor_atividade['Media de dias em OP'] = filtered_df.groupby('Setor de atividade')['Dias em OP'].mean().values.round(0)
    # plot media
    fig = px.bar(setor_atividade, x='Setor de atividade', y='Media de dias em OP', title='Media de dias em OP por setor de atividade')
    st.plotly_chart(fig)

    # qte de itens novos(<30d)
    setor_atividade['Qte de itens'] = filtered_df[filtered_df['Dias em OP'] < dias_filtered].groupby('Setor de atividade')['Dias em OP'].count().values
    # plot qte de itens
    fig = px.bar(setor_atividade, x='Setor de atividade', y='Qte de itens', title='Qte de itens por setor de atividade')
    st.plotly_chart(fig)

    # qte de itens novos com criticidade > 0
    setor_atividade['Qte de itens com criticidade > 0'] = filtered_df[(filtered_df['Dias em OP'] < dias_filtered) & (df['Criticidade'] > 0)].groupby('Setor de atividade')['Dias em OP'].count().values
    # plot
    fig = px.bar(setor_atividade, x='Setor de atividade', y='Qte de itens com criticidade > 0', title='Qte de itens com LMR por setor de atividade')
    st.plotly_chart(fig)

    # valor total dos itens novos
    setor_atividade['Valor total'] = filtered_df[filtered_df['Dias em OP'] < dias_filtered].groupby('Setor de atividade')['Valor total da Orden Planejada'].sum().values.round(0)
    # plot 
    fig = px.bar(setor_atividade, x='Setor de atividade', y='Valor total', title='Valor total por setor de atividade')
    st.plotly_chart(fig)

    # Show data
    st.write(setor_atividade)

with individual_tab:

    # input a specific material
    material = st.text_input('Digite o material', '')
    material_df = df[df['Material'] == material]
    st.table(material_df.T)

    #print reservas in t0028
    reservas = t0028[t0028['Material'] == material]
    st.write('Reservas')
    st.write(reservas)

    #print oc ordenada por data
    oc_df = oc[oc['Material'] == material]
    st.write('Ordens de compra')
    st.write(oc_df.sort_values('Data do documento', ascending=False))

    # plotar consumo por LTD
    # convert 0130 to long format
    t130_long = t130.melt(id_vars='Material', var_name='LTD', value_name='Consumo')
    t130_long = t130_long[t130_long['Material'] == material]
    #limit to 12 LTD
    t130_long = t130_long[t130_long['LTD'].isin([f'{i} LTD' for i in range(1,13)])]
    #remove previsao de entrega
    t130_long = t130_long[t130_long['LTD'] != 'Prz.entrg.prev.']
    fig = px.bar(t130_long, x='LTD', y='Consumo', title=f'Consumo por LTD {df[["Prz.entrg.prev."]].values[0]}')
    # add a line for ponto de reabastecimento
    fig.add_hline(y=material_df['Pt.reabast.'].values[0], line_dash='dot', line_color='red', annotation_text='Ponto de reabastecimento')
    # add a line for estoque maximo
    fig.add_hline(y=material_df['Est.máximo'].values[0], line_dash='dot', line_color='green', annotation_text='Estoque máximo')
    st.plotly_chart(fig)
    
    
print('Fim do script')