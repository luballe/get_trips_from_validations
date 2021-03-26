from google.cloud import bigquery
import pandas as pd
import time
import numpy as np

# Construct a BigQuery client object.
client = bigquery.Client()
start_date='2020-12-05'
end_date='2020-12-05'

# Construct the query
query = """
SELECT 
    A.Numero_Tarjeta as NumeroTarjeta, 
    A.Nombre_Perfil as TipoTarjeta, 
    A.Valor as Valor,
    A.Fecha_Transaccion as Fecha
FROM `transmilenio-dwh-shvpc.validaciones.validacion` A 
WHERE 
A.Fecha_Clearing between '"""+start_date+"""' and '"""+end_date+"""'
ORDER BY NumeroTarjeta, A.Fecha_Transaccion

"""

# API request
query_job = client.query(query)  
# Empty list
rows_list = []
for row in query_job:
    d = {
        'NumeroTarjeta': row['NumeroTarjeta'],
        'TipoTarjeta': row['TipoTarjeta'],
        'Valor': row['Valor'],
        'Fecha': row['Fecha']
    }
    #print(d)
    rows_list.append(d)

df = pd.DataFrame(rows_list)

# Create a copy of the dataframe, shift it one position down and paste it at the right of the original
df_copy = df.copy()
df = df.shift(1)
df = pd.concat([df,df_copy], axis=1)

df = df.reset_index(drop=True)

df.columns = ['NumeroTarjeta_1','TipoTarjeta_1','Valor_1','Fecha_1','NumeroTarjeta_2','TipoTarjeta_2','Valor_2','Fecha_2']

# Generate result using pandas
#minutes = []
ids_viajes= []
first_val_time=pd.NaT
    
for row in df.itertuples():
#    if(row.NumeroTarjeta_1 is not np.nan):
    if(row.NumeroTarjeta_1 == row.NumeroTarjeta_2 and row.Fecha_1.date() == row.Fecha_2.date()):
        mins = (row.Fecha_2-first_val_time).seconds//60
        if (mins<90):
            id_viaje=id_viaje
        else:
            id_viaje=id_viaje+1
            first_val_time=row.Fecha_2  
        #minutes.append(mins) 
        ids_viajes.append(id_viaje)   
    else:
        id_viaje=1
        #minutes.append(0) 
        ids_viajes.append(id_viaje)   
        first_val_time=row.Fecha_2   
        
#df_3 = pd.DataFrame(minutes)
df_temp = pd.DataFrame(ids_viajes)

df = pd.concat([df,df_temp], axis=1)

df.drop(['NumeroTarjeta_1','TipoTarjeta_1', 'Valor_1','Fecha_1'], axis=1, inplace=True)

df.columns = ['NumeroTarjeta','TipoTarjeta','Valor','Fecha','id_viaje']

# Do the grouping and aggregations
df = df.groupby(
   ['NumeroTarjeta', 'TipoTarjeta','id_viaje']
).agg(
    {
         'Valor':sum,    # Sum duration per group
    }
)

df.to_csv('out.csv', header=False)
