from google.cloud import bigquery
from numba import jit, njit, types, vectorize, prange
import os
import gc
import time
import math
import array as arr
import pandas as pd
import numpy as np
import multiprocessing

@njit(nogil=True,fastmath=True)
def calc_viajes(df_arr,job_num,sub_index_arr,last_trip_id):
    #print(job_num)
    #print(sub_index_arr)
    start_index=sub_index_arr[job_num][0]
    end_index=sub_index_arr[job_num][1]
    #print(start_index,end_index)
    num_elems = end_index - start_index
#     print(num_elems)
    indexes       = np.zeros(num_elems,dtype=np.int64)
    ordenes_viaje = np.zeros(num_elems,dtype=np.int32)
    ids_viaje     = np.zeros(num_elems,dtype=np.int64)

#     print(len(indexes))
    i=0
    global_index=start_index
    #for i, global_index in zip(range(0, num_elems), range(start_index, end_index)):
    orden_viaje = 1
    id_viaje    = last_trip_id
    if(num_elems>0):
#         print('-')
#         print(global_index)
#         print(end_index)
        while global_index<end_index:
            indexes[i] = global_index
            NumeroTarjeta_1 = df_arr[global_index][0]
            Epoch_1         = df_arr[global_index][1]
            NumeroTarjeta_2 = df_arr[global_index][2]
            Epoch_2         = df_arr[global_index][3]

            if NumeroTarjeta_1 != NumeroTarjeta_2 :
                id_viaje = id_viaje + 1
                orden_viaje = 1

            else:
                mins = (Epoch_2-Epoch_1)/60
                # 2019: (epoch > 1546300799 and epoch <= 1577836799)
                if (Epoch_2>1546300799 and Epoch_2<=1577836799):
                    if mins>95:
                        id_viaje = id_viaje + 1
                        orden_viaje = 1
                    else:
                        orden_viaje = orden_viaje + 1
                # 2020: (epoch > 1577836799 and epoch <= 1609459199)                    
                elif (Epoch_2>1577836799 and Epoch_2<=1609459199):
                    if mins>110:
                        id_viaje = id_viaje + 1
                        orden_viaje = 1
                    else:
                        orden_viaje = orden_viaje + 1
                else:
                    if mins>90:
                        id_viaje = id_viaje + 1
                        orden_viaje = 1
                    else:
                        orden_viaje = orden_viaje + 1

            ordenes_viaje[i] = orden_viaje
            ids_viaje[i] = id_viaje
            i = i+1
            global_index = global_index+1
    #print(indexes)
        
    return indexes, ordenes_viaje, ids_viaje

@njit(nogil=True,fastmath=True)
def calc_viajes_wrapper(num_proc,df_arr,last_trip_id):
    num_processors = num_proc
    #print("num_processors:",num_processors)
    num_positions=len(df_arr)
    #print("num_positions:",num_positions)
    if num_positions > num_processors:
        block_size=int(math.floor(num_positions/num_processors))
        #print(block_size)

        sub_index_arr= np.zeros((num_processors,2),dtype=np.int32)
        result_arr= np.zeros((num_positions,2),dtype=np.int32)

        start_index = 0
        end_index = block_size

        for i in range(num_processors):
            sub_index_arr[i][0] = int(start_index)
            sub_index_arr[i][1] = int(end_index)
            start_index = start_index + block_size 
            end_index = end_index + block_size
            
        #a = num_positions%num_processors
        #if ((a) > 0):
        sub_index_arr[num_processors-1][1] = num_positions
        
        # Adjusting indexes to match the end of a card number block
        for i in range(num_processors-1):
            index_to_adjust=0
            last_index=sub_index_arr[i][1]
            #df_arr[last_index][2] = NumeroTarjeta_2
            while (df_arr[last_index][2] == df_arr[last_index+index_to_adjust+1][2]):
                index_to_adjust = index_to_adjust + 1
            sub_index_arr[i][1] = sub_index_arr[i][1] + index_to_adjust
            sub_index_arr[i+1][1] = sub_index_arr[i+1][1] + index_to_adjust
            
    else:
        start_index = 0
        end_index = num_positions

        sub_index_arr= np.zeros((1,2),dtype=np.int32)
        result_arr= np.zeros((num_positions,2),dtype=np.int32)
        sub_index_arr[0][0] = 0
        sub_index_arr[0][1] = num_positions
        num_processors = 1
    
    #print(num_processors)
    #print(sub_index_arr)
    #print(block_size)
    
    for job_num in range(num_processors):
        index_arr_res,ordenes_viajes_arr_res,ids_viajes_arr_res = calc_viajes(df_arr,job_num,sub_index_arr,last_trip_id)
        #print(len(index_arr_res))
        for i in range(len(index_arr_res)):
            #result_arr[index_arr_res[i]][0]=index_arr_res[i]
            result_arr[index_arr_res[i]][0]=ordenes_viajes_arr_res[i]
            result_arr[index_arr_res[i]][1]=ids_viajes_arr_res[i]

    return result_arr

def process_query(df,query_index,num_processors,last_trip_id):

    #print(df)
    # Create a copy of the dataframe, shift it one position down and paste it at the right of the original
    df_copy = df[['NumeroTarjeta','Epoch']].copy()
    #print(df_copy)
    #df_copy.dtypes
    df_shifted = df_copy.shift(1, fill_value=0)
    #print(df_shifted)
    
    #df.dtypes
    df_to_process = pd.concat([df_shifted,df_copy], axis=1)
    df_to_process = df_to_process.reset_index(drop=True)

    #print(df_to_process)
    del df_copy
    del df_shifted
    
    #df.columns = ['NumeroTarjeta_1','TipoTarjeta_1','Valor_1','Fecha_1','NumeroTarjeta_2','TipoTarjeta_2','Valor_2','Fecha_2']
    # Define numpy array for numba processing
    #df_arr = df[['NumeroTarjeta_1','Fecha_1','NumeroTarjeta_2','Fecha_2']].to_numpy(dtype=np.int64)
    df_arr = df_to_process.to_numpy(dtype=np.int64)
    del df_to_process
    #print(df_arr)

    # compile calc_viajes
    # array of two positions
    a = np.zeros((0, 2),dtype=np.int32)
    #print('df_arr:')
    #print(df_arr)
    print('Compiling calc_viajes')
    _ = calc_viajes(df_arr,0,a,0)
    print('calc_viajes Compiled!')
    
    start_time = time.time()
    #print('Calculating orders and ids...')
    # calculate orders and ids
    orders_ids_viajes=pd.DataFrame(calc_viajes_wrapper(num_processors,df_arr,last_trip_id),columns=['Orden_viaje','Viaje_id'])
    del(df_arr)
    gc.collect()

    last_trip_id = orders_ids_viajes.iloc[-1]["Viaje_id"]
    print("Last Id:",last_trip_id)

    print('Orders trips and its ids calculated! --- %s seconds ---' % (time.time() - start_time))
    
    df = pd.concat([df,orders_ids_viajes], axis=1)
    #print(df)
    #df.drop(['NumeroTarjeta_1','TipoTarjeta_1', 'Valor_1','Fecha_1'], axis=1, inplace=True)
    df.drop(['Epoch'], axis=1, inplace=True)
    #df.columns = ['NumeroTarjeta','TipoTarjeta','Valor','Fecha','Orden_viaje','Viaje_id']
    
    query_index=query_index+1
    filename="detalles_viajes_"+str('{:02d}'.format(query_index))+".csv".zfill(3)
    print("Saving file "+filename+"...")
    df.to_csv(filename,index=False,header=False)
    print("File "+filename+" saved!")
  
    
    #print(df)
    # Do the grouping and aggregations
#     df = df.groupby(["Viaje_id"])["Valor"].sum()
    df = df.groupby(
       ['Viaje_id']
    ).agg(
        {
            'Valor':sum,    # Sum duration per group
        }
    )
    df = df.astype({'Valor': 'int32'})
    #print(df)
    filename="resumen_viajes_"+str('{:02d}'.format(query_index))+".csv".zfill(3)
    print("Saving file "+filename+"...")
    df.to_csv(filename,index=True,header=False)
    print("File "+filename+" saved!")

    # Consolidate results
    print("Consolidating results...")
    os.popen('cat detalles_viajes_'+str('{:02d}'.format(query_index))+'.csv >> detalles.csv')
    os.popen('rm detalles_viajes_'+str('{:02d}'.format(query_index))+'.csv')

    os.popen('cat resumen_viajes_'+str('{:02d}'.format(query_index))+'.csv >> resumen.csv')
    os.popen('rm resumen_viajes_'+str('{:02d}'.format(query_index))+'.csv')
    print("Results consolidated!")
    
    
    return last_trip_id

if __name__ == '__main__':
    #num_processors=multiprocessing.cpu_count()
    num_processors=1
    #print('Num Processors: ',num_processors)
    # Construct a BigQuery client object.
    client = bigquery.Client()
    print("Getting trips of cards ...")
    dates=[
        ['2019-09-01','2019-09-30'],
        ['2020-02-01','2020-02-29'],
        ['2020-10-01','2020-10-31'],
          ]
    # Construct the query
    query = """

    SELECT distinct Numero_Tarjeta, count(Fecha_Clearing) as Num_val
    FROM `transmilenio-dwh-shvpc.validaciones.validacion` 
    WHERE (
        (Fecha_Clearing between '"""+dates[0][0]+"""' and '"""+dates[0][1]+"""') or 
        (Fecha_Clearing between '"""+dates[1][0]+"""' and '"""+dates[1][1]+"""') or 
        (Fecha_Clearing between '"""+dates[2][0]+"""' and '"""+dates[2][1]+"""') 
    ) 
    GROUP BY Numero_Tarjeta
    ORDER BY Numero_Tarjeta 

    """
    
    # API request
    df = client.query(query).to_dataframe(
        dtypes={"Numero_Tarjeta": "int64",
                "Num_val": "int32"})
    print("Cards gotten!")
    
    #Let's create and array of card numbers. Each row 
    #contains a range of card number which count of validation is about 10M 

    #index_array = []
    card_num_array = []
    ini=0
    end=0
    accum=0
    for row in df.itertuples():
        if accum > 10000000:
            #index_array.append((ini, end))
            card_num_array.append((df.iloc[ini]["Numero_Tarjeta"], df.iloc[end]["Numero_Tarjeta"]))
            accum=0
            ini=end
            #end=0

        accum = accum + row.Num_val
        end=end+1
    #index_array.append((ini, end-1))
    card_num_array.append((df.iloc[ini]["Numero_Tarjeta"], df.iloc[end-1]["Numero_Tarjeta"]))
    #card_num_array = np.vstack(index_array)
    #print(card_num_array)
    #print(len(card_num_array))
    
    print("Processing batches of cards...")
    arr_len = len(card_num_array)
    #arr_len = 2
    last_trip_id = 0
    for i in range(arr_len):
        
        print("Card Range: ",card_num_array[i][0],card_num_array[i][1])
        # Construct the query
        query = """
        SELECT 
            A.Numero_Tarjeta as NumeroTarjeta,
            SUBSTR(Nombre_Perfil , 2, 3) as TipoTarjeta, 
            A.Valor as Valor,
            Fecha_Transaccion as Fecha,
            UNIX_SECONDS(A.Fecha_Transaccion) as Epoch
        FROM `transmilenio-dwh-shvpc.validaciones.validacion` A 
        WHERE 
            (
                (Fecha_Clearing between '"""+dates[0][0]+"""' and '"""+dates[0][1]+"""') or 
                (Fecha_Clearing between '"""+dates[1][0]+"""' and '"""+dates[1][1]+"""') or 
                (Fecha_Clearing between '"""+dates[2][0]+"""' and '"""+dates[2][1]+"""') 
            )  and 
            (
                cast(A.Numero_Tarjeta as int64) >= """ + str(card_num_array[i][0]) + """ and 
                cast(A.Numero_Tarjeta as int64) < """ + str(card_num_array[i][1]) + """
            )
        ORDER BY NumeroTarjeta, Fecha

        """
        #print(query)
        # API request
        df = client.query(query).to_dataframe(
            dtypes={"NumeroTarjeta": "int64",
                    "TipoTarjeta": "int8",
                    "Valor":"int16",
                    "Fecha":"datetime64",
                    "Epoch":"int64"})
        df["Fecha"]=df["Fecha"].dt.strftime("%Y-%m-%d %H:%M:%S")
        #print(df)
        last_trip_id = process_query(df,i,num_processors,last_trip_id)
    print("All Done!")
