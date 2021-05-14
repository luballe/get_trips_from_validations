from google.cloud import bigquery
from numba import jit, njit, types, vectorize, prange
from multiprocessing import Lock, Process, Queue, current_process
import threading
import multiprocessing
import os
import gc
import csv
import time
import math
import queue
import subprocess
import array as arr
import pandas as pd
import numpy as np
import datetime as dt

@njit(nogil=True,fastmath=True)
def calc_viajes(df_arr,job_num,sub_index_arr,last_trip_id,ini_epoch,end_epoch,minutes_trip):
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
    if(num_elems>0):
#         print('-')
#         print(global_index)
#         print(end_index)
        i=0
        global_index=start_index
        #for i, global_index in zip(range(0, num_elems), range(start_index, end_index)):
        orden_viaje = 1
        id_viaje    = last_trip_id
        elapsed_time = 0 
        while global_index<end_index:
            indexes[i] = global_index
            NumeroTarjeta_1 = df_arr[global_index][0]
            Epoch_1         = df_arr[global_index][1]
            NumeroTarjeta_2 = df_arr[global_index][2]
            Epoch_2         = df_arr[global_index][3]

            if NumeroTarjeta_1 != NumeroTarjeta_2 :
                id_viaje = id_viaje + 1
                orden_viaje = 1
                elapsed_time = 0
            else:
                mins = (Epoch_2-Epoch_1)/60
                elapsed_time = elapsed_time + mins
                #if (Epoch_2>ini_epoch and Epoch_2<=end_epoch):
                if elapsed_time<=minutes_trip:
                    orden_viaje = orden_viaje + 1
                else:
                    id_viaje = id_viaje + 1
                    orden_viaje = 1
                    elapsed_time = 0

            ordenes_viaje[i] = orden_viaje
            ids_viaje[i] = id_viaje
            #elapsed_times[i] = elapsed_time
            #filter_valid[i] = valid
            i = i+1
            global_index = global_index+1
    #print(indexes)
        
    #return indexes, ids_viaje, ordenes_viaje, elapsed_times
    return indexes, ids_viaje, ordenes_viaje

@njit(nogil=True,fastmath=True)
def calc_viajes_wrapper(num_proc,df_arr,last_trip_id,ini_epoch,end_epoch,minutes_trip):
    num_processors = num_proc
    #print("num_processors:",num_processors)
    num_positions=len(df_arr)
    #print("num_positions:",num_positions)
    if num_positions > num_processors:
        block_size=int(math.floor(num_positions/num_processors))
        #print(block_size)

        sub_index_arr= np.zeros((num_processors,2),dtype=np.int32)
        #result_arr= np.zeros((num_positions,3),dtype=np.int32)
        # No elapsed times
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
        #index_arr_res,ids_viajes_arr_res,ordenes_viajes_arr_res,elapsed_times_arr_res = calc_viajes(df_arr,job_num,sub_index_arr,last_trip_id)
        index_arr_res,ids_viajes_arr_res,ordenes_viajes_arr_res = calc_viajes(df_arr,job_num,sub_index_arr,last_trip_id,ini_epoch,end_epoch,minutes_trip)
        #print(len(index_arr_res))
        for i in range(len(index_arr_res)):
            #result_arr[index_arr_res[i]][0]=index_arr_res[i]
            result_arr[index_arr_res[i]][0]=ids_viajes_arr_res[i]
            result_arr[index_arr_res[i]][1]=ordenes_viajes_arr_res[i]
            #result_arr[index_arr_res[i]][2]=elapsed_times_arr_res[i]
            #result_arr[index_arr_res[i]][3]=filter_valid_arr_res[i]
            

    return result_arr
def process_query(df,query_index,num_processors,last_trip_id,date_info):

    ini_date=date_info[0]
    end_date=date_info[1]
    month=date_info[2]
    ini_epoch=date_info[3]
    end_epoch=date_info[4]
    minutes_trip=date_info[5]
    max_validations=date_info[6]
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
    
    # Define numpy array for numba processing
    #df_arr = df[['NumeroTarjeta_1','Fecha_1','NumeroTarjeta_2','Fecha_2']].to_numpy(dtype=np.int64)
    df_arr = df_to_process.to_numpy(dtype=np.int64)
    del df_to_process
    #print(df_arr)
    

    # compile calc_viajes
    # array of two positions
    a = np.zeros((0, 2),dtype=np.int32)
    #print('Compiling calc_viajes')
    _ = calc_viajes(df_arr,0,a,0,0,0,0)
    #print('calc_viajes Compiled!')
    
    start_time = time.time()
    print("Proc "+str(query_index)+" Calculating details (trip_IDs and trip_order)...")
    # calculate orders and ids
    orders_ids_viajes=pd.DataFrame(calc_viajes_wrapper(num_processors,df_arr,last_trip_id,ini_epoch,end_epoch,minutes_trip),columns=['Viaje_id','Orden_viaje'])
    #print(orders_ids_viajes)
    del(df_arr)
    gc.collect()

    last_trip_id = orders_ids_viajes.iloc[-1]["Viaje_id"]
    #print("Last Id:",last_trip_id)

    print("Proc "+str(query_index)+" Orders trips and its ids calculated! --- %s seconds ---" % (time.time() - start_time))
    
    df = pd.concat([df,orders_ids_viajes], axis=1)

    #print(df)

    df.drop(['Epoch'], axis=1, inplace=True)
    
    #query_index=query_index+1
    filename="detalles_"+str(month)+"_"+str('{:02d}'.format(query_index))+".csv".zfill(3)
    print("Proc "+str(query_index)+" Saving file "+filename+"...")
    df.to_csv(filename,index=False)
    print("Proc "+str(query_index)+" File "+filename+" saved!")
    
    return last_trip_id

def process_card_batch(i,ini_card,end_card,date_info):
    i=i+1
    print("Proc "+str(i)+" Getting cards info. Card's range: "+str(ini_card)+"-"+str(end_card))
    # Construct the query
    ini_date=date_info[0]
    end_date=date_info[1]
    month=date_info[2]

    query = """
    SELECT 
        A.Numero_Tarjeta as NumeroTarjeta,
        SUBSTR(Nombre_Perfil , 2, 3) as TipoTarjeta, 
        A.Valor as Valor,
        Fecha_Transaccion as Fecha,
        UNIX_SECONDS(A.Fecha_Transaccion) as Epoch
    FROM `transmilenio-dwh-shvpc.validaciones.validacion` A 
    WHERE 
        Fecha_Clearing between '"""+ini_date+"""' and '"""+end_date+"""' and 
        (
            cast(A.Numero_Tarjeta as int64) >= """ + str(ini_card) + """ and 
            cast(A.Numero_Tarjeta as int64) < """ + str(end_card) + """
        )
    ORDER BY NumeroTarjeta, Fecha
    """
    #print(query)
    # Create a instance of bigquery 
    client = bigquery.Client()
    # API request
    df = client.query(query).to_dataframe(
        dtypes={"NumeroTarjeta": "int64",
                "TipoTarjeta": "int8",
                "Valor":"int16",
                "Fecha":"datetime64",
                "Epoch":"int64"})
    df["Fecha"]=df["Fecha"].dt.strftime("%Y-%m-%d %H:%M:%S")
    #print(df.dtypes)

    #print(df)
    print("Proc "+str(i)+" Info received!")
    # Extracción del mes y el día (para conservarlos después)
    #df['mes'] = pd.to_datetime(df["Fecha"]).dt.to_period('M')
    print("Proc "+str(i)+" Adding month and day to dataframe...")
    df['mes'] = month
    df['dia'] = pd.to_datetime(df["Fecha"]).dt.to_period('D')
    
    last_trip_id = 0
    print("Proc "+str(i)+" Starting process batch of cards...")
    last_trip_id = process_query(df,i,1,last_trip_id,date_info)
    
def do_job_1(tasks_to_accomplish, tasks_that_are_done):
    while True:
        try:
            #try to get task from the queue. get_nowait() function will 
            #raise queue.Empty exception if the queue is empty. 
            #queue(False) function would do the same task also.
            task = tasks_to_accomplish.get_nowait()
            
        except queue.Empty:

            break
        else:
            #if no exception has been raised, add the task completion 
            #message to task_that_are_done queue
            #print(task)
            i=task[0]
            id_batch = i + 1
            num_batches = task[1]
            ini_card = task[2]
            end_card = task[3]
            date_info = task[4]
            
            print(current_process().name+" Processing batch "+str(id_batch)+" of "+str(num_batches)+ "\n")
            # Call the function that gets details, viajes, dia and mes
            process_card_batch(task[0],ini_card,end_card,date_info)
            tasks_that_are_done.put(task)
            #time.sleep(1)
    return True

def process_date(num_processors,validations_batch_size,date_info):
    #print(date_info)
    ini_date=date_info[0]
    end_date=date_info[1]
    month=date_info[2]
    max_validations=date_info[6]
    # Construct the query
    query = """
    SELECT distinct Numero_Tarjeta, count(Fecha_Clearing) as Num_val
    FROM `transmilenio-dwh-shvpc.validaciones.validacion` 
    WHERE (
        (Fecha_Clearing between '"""+ini_date+"""' and '"""+end_date+"""') 
    ) 
    --AND cast(Numero_Tarjeta as int64) in ( """ + str(1010000002769891) + """, """ +str(1010000002770386)+""")  
    
    GROUP BY Numero_Tarjeta
    ORDER BY Numero_Tarjeta 
    """
    # Create a instance of bigquery 
    client = bigquery.Client()
    # API request
    df = client.query(query).to_dataframe(
        dtypes={"Numero_Tarjeta": "int64",
                "Num_val": "int32"})
    print("Cards gotten!")

    #Let's create and array of card numbers. Each row contains a range of cards numbers.
    # Count of validations of about validations_batch_size (~5M) Validations
    print("Distributing validations in batches of "+str(validations_batch_size)+" records..")
    #index_array = []
    card_num_array = []
    ini=0
    end=0
    accum=0
    for row in df.itertuples():
        if accum > validations_batch_size:
            card_num_array.append((df.iloc[ini]["Numero_Tarjeta"], df.iloc[end]["Numero_Tarjeta"]))
            accum=0
            ini=end

        accum = accum + row.Num_val
        end=end+1
    card_num_array.append((df.iloc[ini]["Numero_Tarjeta"], df.iloc[end-1]["Numero_Tarjeta"]))
    #print(card_num_array)
    #print(len(card_num_array))
    print("Load distributed!")
    print(card_num_array)
    
    print("Processing batches of cards...")
    arr_len = len(card_num_array)
        
    # Multiprocessing - taking advantage of all computer processors
    number_of_task = arr_len
    number_of_processes = num_processors
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []
    
    for i in range(number_of_task):
        ini_card=card_num_array[i][0]
        end_card=card_num_array[i][1]
        #index=i+1
        elem=(i,number_of_task,ini_card,end_card,date_info)
        #print(elem)
        tasks_to_accomplish.put(elem)

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=do_job_1, args=(tasks_to_accomplish, tasks_that_are_done))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    print("Batches that are done:")
    while not tasks_that_are_done.empty():
        print(tasks_that_are_done.get())
    
    # Consolidating
    print("Consolidating sub-results by month...")
    start_time = time.time()
    
    # Consolidating detalles and viajes
    print("Consolidating detalles ...")
    
    offset_id_viaje = 0
    df_details = pd.DataFrame()
    files_detalles_str=""
    for i in range(number_of_task):
        i=i+1
        # Details
        filename="detalles_"+str(month)+"_"+str('{:02d}'.format(i))+".csv".zfill(3)
        print("Concatenating "+str(i)+" "+filename+"...")
        df = pd.read_csv(filename)
        if(i == 1):
            offset_id_viaje = df['Viaje_id'].iloc[-1] 
        else:
            df['Viaje_id'] = df['Viaje_id'] + offset_id_viaje
            offset_id_viaje = df['Viaje_id'].iloc[-1]
        df_details = pd.concat([df_details, df],ignore_index=True)
        files_detalles_str = files_detalles_str + filename + " "
    del df
    gc.collect()
    
    print("Calculating viajes...")
    # Group by - por tarjeta, fecha y viaje (agrupa a nivel de viaje). Deja el último de tarjeta que aparezca en el viaje
    df_viajes = df_details.groupby(['NumeroTarjeta','mes','dia','Viaje_id']).agg(
            tipo_tarjeta=pd.NamedAgg(column='TipoTarjeta', aggfunc='last'),
            valor=pd.NamedAgg(column='Valor', aggfunc=np.nansum),
            validaciones = pd.NamedAgg(column='NumeroTarjeta', aggfunc='size')).reset_index()
    
    print("Calculating info_tarjeta_dia...")
    df_itd = df_viajes.loc[( (df_viajes['validaciones'] <= max_validations) & (df_viajes['mes'] == str(month)) )]

    df_itd = df_itd.groupby(['NumeroTarjeta','mes','dia']).agg(
        tipo_tarjeta=pd.NamedAgg(column='tipo_tarjeta', aggfunc='last'),
        valor_dia=pd.NamedAgg(column='valor', aggfunc=np.nansum),
        viajes_dia = pd.NamedAgg(column='Viaje_id', aggfunc='size'),
        validaciones_dia = pd.NamedAgg(column='validaciones', aggfunc=np.nansum)).reset_index()

    print("Calculating info_tarjeta_mes...")
    df_itm = df_itd.groupby(['NumeroTarjeta','mes']).agg(
        tipo_tarjeta = pd.NamedAgg(column='tipo_tarjeta', aggfunc='last'),
        valor_mes = pd.NamedAgg(column='valor_dia', aggfunc=np.nansum),
        viajes_mes = pd.NamedAgg(column='viajes_dia', aggfunc=np.nansum),
        validaciones_mes = pd.NamedAgg(column='validaciones_dia', aggfunc=np.nansum),
        p_valor_dia = pd.NamedAgg(column='valor_dia', aggfunc=np.nanmean),
        valor_dia_min = pd.NamedAgg(column='valor_dia', aggfunc='min'),
        valor_dia_max = pd.NamedAgg(column='valor_dia', aggfunc='max'),
        p_viajes_dia = pd.NamedAgg(column='viajes_dia', aggfunc=np.nanmean),
        viajes_dia_min = pd.NamedAgg(column='viajes_dia', aggfunc='min'),
        viajes_dia_max = pd.NamedAgg(column='viajes_dia', aggfunc='max'),
        p_valida_dia = pd.NamedAgg(column='validaciones_dia', aggfunc=np.nanmean),
        valida_dia_min = pd.NamedAgg(column='validaciones_dia', aggfunc='min'),
        valida_dia_max = pd.NamedAgg(column='validaciones_dia', aggfunc='max')).reset_index()

    print("Saving Final CSV Files...")
    
    filename="detalles_"+str(month)+".csv"
    print("Saving file "+filename+"...")
    df_details.to_csv(filename,index=False)
    
    filename="viajes_"+str(month)+".csv"
    print("Saving file "+filename+"...")
    df_viajes.to_csv(filename,index=False)

    filename="info_tarjeta_dia_"+str(month)+".csv"
    print("Saving file "+filename+"...")
    df_itd.to_csv(filename,index=False)
    
    filename="info_tarjeta_mes_"+str(month)+".csv"
    print("Saving file "+filename+"...")
    df_itm.to_csv(filename,index=False)
    
    print("Uploading Results...")
    gdrive_dir_id="1YJo2TClzlXJVyPUv4z6eJYgoqTNI6ndl"
    path="/monitoreo/viajes_tmsa_andrea/"
    #command="/monitoreo/gdrive/gdrive upload -p "+gdrive_dir_id+" "+path+"detalles_"+str(month)+".csv"

    command=["/monitoreo/gdrive/gdrive","upload","-p",gdrive_dir_id,path+"detalles_"+str(month)+".csv"]
    print(command)
    subprocess.Popen(command)

    command=["/monitoreo/gdrive/gdrive","upload","-p",gdrive_dir_id,path+"viajes_"+str(month)+".csv"]
    print(command)
    subprocess.Popen(command)
    
    command=["/monitoreo/gdrive/gdrive","upload","-p",gdrive_dir_id,path+"info_tarjeta_dia_"+str(month)+".csv"]
    print(command)
    subprocess.Popen(command)
    
    command=["/monitoreo/gdrive/gdrive","upload","-p",gdrive_dir_id,path+"info_tarjeta_mes_"+str(month)+".csv"]
    print(command)
    subprocess.Popen(command)
    

    os.popen("rm "+files_detalles_str)

    print('Files Consolidated! --- %s seconds ---' % (time.time() - start_time))
    
#last_trip_id = 0
if __name__ == '__main__':
    total_start_time = time.time()
    num_processors=multiprocessing.cpu_count()
    validations_batch_size=2000000
    os.popen("rm *.csv")

    print("Getting trips info (count of validations for each card) ...")
    dates=[
        ['2019-09-01','2019-09-30','2019-09',1567295999,1569887999,95,3],
        ['2020-02-01','2020-02-29','2020-02',1580515199,1583020799,95,3],
        ['2020-10-01','2020-10-31','2020-10',1601510399,1604188799,110,4],
        
#         ['2019-09-01','2019-09-03','2019-09',1567295999,1569887999,95,3],
#         ['2020-02-01','2020-02-03','2020-02',1580515199,1583020799,95,3],
#         ['2020-10-01','2020-10-03','2020-10',1601510399,1604188799,110,4],
    ]
    
    for date_info in dates:
        date_start_time = time.time()
        print("Processing date: "+str(date_info[2]))
        process_date(num_processors,validations_batch_size,date_info)
        print("* Date " +date_info[2]+" Finished! * --- %s seconds ---" % (time.time() - date_start_time))
    print("*** All Done! *** --- %s seconds ---" % (time.time() - total_start_time))
    
