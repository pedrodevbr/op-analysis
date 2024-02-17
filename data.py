import pandas as pd
import numpy as np

####################
# Ordens de compra #
####################

oc_df = pd.read_excel('oc.XLSX')

# Exclude rows where 'texto breve' contains "anulado" or "cancelado"
oc_df = oc_df[~oc_df['Texto breve'].str.contains('anulado|cancelado', case=False, na=False)]
oc_df['Material'] = oc_df['Material'].astype(str).replace('\.0', '', regex=True)
# Convert 'Data do documento' to datetime
oc_df['Data do documento'] = pd.to_datetime(oc_df['Data do documento'])

# for each material get the date of the last order
last_order_date = oc_df.groupby('Material')['Data do documento'].max().reset_index()

last_order_date['Documento de compras'] = oc_df['Documento de compras'].astype(str)

# group by "material" all the fornecedores
fornecedores = oc_df.groupby('Material')['Fornecedor/centro fornecedor'].apply(list).reset_index()
print(fornecedores.head()) 

###############
# Requisições #
###############

req_df = pd.read_excel('req.XLSX')
req_df['Material'] = req_df['Material'].astype(str).replace('\.0', '', regex=True)
req_df['Requisição de compra'] = req_df['Requisição de compra'].astype(str).replace('\.0', '', regex=True)
req_df['Data ultima OC'] = pd.to_datetime(req_df['Data da solicitação'])

# For each material, get the Code of Elimination for the last requisition
# if the last code of elimination is 1, then 
last_req = req_df.sort_values('Data da solicitação').groupby('Material').last().reset_index()

# The resulting DataFrame will have columns for 'Material', 'Data da solicitação', 'Número da requisição', and any other columns in req_df
last_req_date = last_req[['Material', 'Data da solicitação', 'Requisição de compra','Código de eliminação']]
print(last_req_date.head())

# if code of elimination is 1, substitute for eliminado
# and the collum name for Ultima REQ

last_req_date['Código de eliminação'] = last_req_date['Código de eliminação'].replace(1, 'Eliminado')
last_req_date.rename(columns={'Data da solicitação': 'Data ultima REQ', 'Requisição de compra': 'Ultima REQ'}, inplace=True)

###############
# Saneamento  #
###############

san_df = pd.read_excel('san.XLSX')
san_df['Material'] = san_df['Material'].astype(str).replace('\.0', '', regex=True)
lmr_df = san_df[['Material','LMR?']]
#print(lmr_df.head())

# join tables
df = pd.merge(last_order_date, last_req_date, on='Material', how='outer')

# Merge the result with last_order_date
df = pd.merge(df, lmr_df, on='Material', how='left')

print(df.head())

###########
# consumo #
###########

cons_df = pd.read_excel('consumo.XLSX')
cons_df['Material'] = cons_df['Material'].astype(str).replace('\.0', '', regex=True)

# extract the the collumns 3 to 15 and calculate the 2 gradest value
def seg_consLT(x):
    x = x[3:15]
    x = x.sort_values(ascending=False)
    x = x[:2]
    return x

cons_df['seg_consLT'] = cons_df.apply(seg_consLT, axis=1)

#extract the trend of last 4 values
def trend(x):
    x = x[:4]
    x = x.mean()
    return x

cons_df['trend'] = cons_df.apply(trend, axis=1)

print(cons_df.head())

