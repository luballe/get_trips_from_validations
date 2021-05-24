from google.cloud import bigquery
from numba import jit, njit, types, vectorize, prange
from multiprocessing import Lock, Process, Queue, current_process, shared_memory
from multiprocessing.managers import SharedMemoryManager
import threading
import multiprocessing
import os
import gc
import csv
import time
import math
import queue
import subprocess
from subprocess import Popen, PIPE, STDOUT
import array as arr
import pandas as pd
import numpy as np
import datetime as dt

@njit(nogil=True,fastmath=True)
def calc_viajes(np_arr,job_num,sub_index_arr,ini_epoch,end_epoch,minutes_trip):
    #print(job_num)
    #print(sub_index_arr)
    start_index=sub_index_arr[job_num][0]
    end_index=sub_index_arr[job_num][1]
    #print(start_index,end_index)
    num_elems = end_index - start_index
#     print(num_elems)
    indexes       = np.zeros(num_elems,dtype=np.int64)
    ids_viaje     = np.zeros(num_elems,dtype=np.int64)
    ordenes_viaje = np.zeros(num_elems,dtype=np.int16)
    valores_viaje = np.zeros(num_elems,dtype=np.int32)
    flags_viaje   = np.zeros(num_elems,dtype=np.int8)


#     print(len(indexes))
    if(num_elems>0):
#         print('-')
#         print(global_index)
#         print(end_index)
        i=0
        global_index=start_index
        orden_viaje  = 1
        id_viaje     = 0
        elapsed_time = 0 
        valor_accum  = 0
        flag_viaje   = 1
        while global_index<end_index:
            indexes[i] = global_index
            NumeroTarjeta_1 = np_arr[global_index][0]
            TipoTarjeta_1   = np_arr[global_index][1]
            Valor_1         = np_arr[global_index][2]
            Epoch_1         = np_arr[global_index][3]

            NumeroTarjeta_2 = np_arr[global_index][4]
            TipoTarjeta_2   = np_arr[global_index][5]
            Valor_2         = np_arr[global_index][6]
            Epoch_2         = np_arr[global_index][7]

            if NumeroTarjeta_1 != NumeroTarjeta_2 :
                id_viaje = id_viaje + 1
                orden_viaje = 1
                elapsed_time = 0
                valor_accum  = Valor_1
                flag_viaje   = 1
            else:
                mins = (Epoch_2-Epoch_1)/60
                elapsed_time = elapsed_time + mins

                #if (Epoch_2>ini_epoch and Epoch_2<=end_epoch):
                if elapsed_time<=minutes_trip:
                    orden_viaje = orden_viaje + 1
                    flag_viaje  = 0
                    valor_accum = valor_accum + Valor_1
                else:
                    id_viaje = id_viaje + 1
                    orden_viaje = 1
                    elapsed_time = 0
                    valor_accum  = Valor_1
                    flag_viaje   = 1

            ordenes_viaje[i] = orden_viaje
            ids_viaje[i]     = id_viaje
            valores_viaje[i] = valor_accum
            if(i>0):
                flags_viaje[i-1]   = flag_viaje

            i = i+1
            global_index = global_index+1

        # The last element should be flagged
        flags_viaje[end_index-1]   = 1
    #print(indexes)

    return indexes, ids_viaje, ordenes_viaje, valores_viaje, flags_viaje

@njit(nogil=True,fastmath=True)
def calc_viajes_wrapper(num_proc,np_arr,ini_epoch,end_epoch,minutes_trip):
    num_processors = num_proc
    #print("num_processors:",num_processors)
    num_positions=len(np_arr)
    #print("num_positions:",num_positions)
    if num_positions > num_processors:
        block_size=int(math.floor(num_positions/num_processors))
        #print(block_size)

        sub_index_arr= np.zeros((num_processors,2),dtype=np.int32)
        # 4 positions for trip_id, order, value_accum and flags
        result_arr= np.zeros((num_positions,4),dtype=np.int32)

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
            #np_arr[last_index][2] = NumeroTarjeta_2
            while (np_arr[last_index][2] == np_arr[last_index+index_to_adjust+1][2]):
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
        index_arr_res,ids_viajes_arr_res,ordenes_viajes_arr_res,valores_viaje_res,flags_viaje_res = calc_viajes(np_arr,job_num,sub_index_arr,ini_epoch,end_epoch,minutes_trip)
        #print(len(index_arr_res))
        for i in range(len(index_arr_res)):
            #result_arr[index_arr_res[i]][0]=index_arr_res[i]
            result_arr[index_arr_res[i]][0] = ids_viajes_arr_res[i]
            result_arr[index_arr_res[i]][1] = ordenes_viajes_arr_res[i]
            result_arr[index_arr_res[i]][2] = valores_viaje_res[i]
            result_arr[index_arr_res[i]][3] = flags_viaje_res[i]
            

    return result_arr

def process_query(np_arr,id_batch,num_processors,date_info):

    ini_date=date_info[0]
    end_date=date_info[1]
    month=date_info[2]
    ini_epoch=date_info[3]
    end_epoch=date_info[4]
    minutes_trip=date_info[5]
    max_validations=date_info[6]
    
#     print("np_arr shape: "+str(np_arr.shape))
    # Create a copy of the original array with the card number (column 0), TipoTarjeta (Column 1), Valor (Column 2) and epoch (column 3)
    np_arr_copy = np_arr[:,[0,1,2,3]]
#     print("np_arr_copy shape: "+str(np_arr_copy.shape))
    # Shift the array copied one position down
    np_arr_copy = np.roll(np_arr_copy, 1,axis=0)
#     print(np_arr_copy)
    # Concatenate the copied array to the right of the original one (just with the card number (column 0) and epoch (column 3))
    np_arr_tmp = np.concatenate((np_arr[:,[0,1,2,3]], np_arr_copy), axis=1)
    del np_arr_copy
    gc.collect()
#     print(np_arr_tmp)

    # compile calc_viajes
    start_time = time.time()
    print("Task "+str(id_batch)+" Compiling calc_viajes...")
    # array of two positions
    a = np.zeros((0, 2),dtype=np.int32)
    #print('Compiling calc_viajes')
    _ = calc_viajes(np_arr_tmp,0,a,0,0,0)
    #print('calc_viajes Compiled!')
    print("Task "+str(id_batch)+" calc_viajes compiled in --- %s seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    print("Task "+str(id_batch)+" Calculating details (trip_IDs and trip_order)...")
    # calculate orders and ids
    trips_results=calc_viajes_wrapper(num_processors,np_arr_tmp,ini_epoch,end_epoch,minutes_trip)
#     print("trips_results shape: "+str(trips_results.shape))
#     print(trips_results)
    del(np_arr_tmp)

    # Calculate las trip id
    last_trip_id = trips_results[-1,0]
#     print("Last Id:",last_trip_id)

    print("Task "+str(id_batch)+" Trips_results calculated! --- %s seconds ---" % (time.time() - start_time))
    
    # Concatenate results to return.
    np_arr = np.concatenate((np_arr, trips_results), axis=1)
#     print("np_arr shape: "+str(np_arr.shape))
#     print(np_arr)

    return last_trip_id,np_arr


def process_card_batch(id_batch,ini_card,end_card,date_info):
#     i=i+1
    print("Task "+str(id_batch)+" Querying BigQuery. Card's range: "+str(ini_card)+"-"+str(end_card))
    # Construct the query
    ini_date=date_info[0]
    end_date=date_info[1]
    month=date_info[2]

    query = """
    SELECT 
        A.Numero_Tarjeta as NumeroTarjeta,
        SUBSTR(Nombre_Perfil , 2, 3) as TipoTarjeta, 
        A.Valor as Valor,
        UNIX_SECONDS(A.Fecha_Transaccion) as Epoch
    FROM `transmilenio-dwh-shvpc.validaciones.validacion` A 
    WHERE 
        ( Fecha_Clearing between DATE '"""+ini_date+"""' and DATE_ADD( DATE '"""+end_date+"""', INTERVAL 15 DAY) )
        and ( Fecha_Transaccion >= TIMESTAMP '"""+ini_date+""" 00:00:00') and (Fecha_Transaccion <= TIMESTAMP '"""+end_date+""" 23:59:59')
        and (
            cast(A.Numero_Tarjeta as int64) >= """ + str(ini_card) + """ and 
            cast(A.Numero_Tarjeta as int64) < """ + str(end_card) + """
        )
    ORDER BY NumeroTarjeta, Epoch
    """
    #print(query)
    # Create a instance of bigquery 
    client = bigquery.Client()
    # API request
    df = client.query(query).to_dataframe(
        dtypes={"NumeroTarjeta": "int64",
                "TipoTarjeta": "int8",
                "Valor":"int16",
                "Epoch":"int64"})
#     df["Fecha"]=df["Fecha"].dt.strftime("%Y-%m-%d %H:%M:%S")
    #print(df.dtypes)

    #print(df)
    print("Task "+str(id_batch)+" results received from Google BigQuery!")
    

    print("Task "+str(id_batch)+" Starting process batch of cards...")
#     dtype=[
#         ('NumeroTarjeta', '<i8'), 
#         ('TipoTarjeta', 'i1'), 
#         ('Valor', '<i2'), 
#         ('Epoch', '<i8')]
    np_arr = df.to_numpy()
#     a = a.astype(dtype)
#     print(a.dtype)
    return process_query(np_arr,id_batch,1,date_info)
    
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
            start_time = time.time()

            i           = task[0]
            num_batches = task[1]
            ini_card    = task[2]
            end_card    = task[3]
            date_info   = task[4]
            shm1_name   = task[5]
            shm2_name   = task[6]

            id_batch    = i + 1
            
            print(current_process().name+" Processing task "+str(id_batch)+" of "+str(num_batches)+ "\n")
            # Call the function that gets details, viajes, dia and mes
            last_trip_id,np_arr = process_card_batch(id_batch,ini_card,end_card,date_info)
#             print(np_arr)
#             np_format = ['%i', '%i', '%i', '%i', '%i', '%i', '%i', '%i']
#             header    = "NumeroTarjeta,TipoTarjeta,Valor,Fecha,Viaje_id,Orden,Valor_Accum,Flag"
#             np.savetxt("test.csv",np_arr,delimiter=",",fmt=np_format,header=header,comments="")
            
            NumeroTarjeta = np_arr[:,0]
            TipoTarjeta   = np_arr[:,1]
            Valor         = np_arr[:,2]
            Fecha         = np_arr[:,3]
            Viaje_id      = np_arr[:,4]
            Order         = np_arr[:,5]
            Valor_Accum   = np_arr[:,6]
            Flag          = np_arr[:,7]

            details_dtype=[
                ('NumeroTarjeta', '<i8'),
                ('TipoTarjeta', '<i1'),
                ('Valor', '<i2'),
                ('Fecha', '<M8[s]'),
                ('Viaje_id', '<i8'),
                ('Order', '<i2'),
            ]
#             names=('NumeroTarjeta', 'TipoTarjeta', 'Valor', 'Fecha', 'Viaje_id', 'Order', 'Valor_accum', 'Flag')
            names=('NumeroTarjeta', 'TipoTarjeta', 'Valor', 'Fecha', 'Viaje_id', 'Order')
            details_arr = np.rec.fromarrays(
#                 arrayList=[NumeroTarjeta, TipoTarjeta, Valor, Fecha, Viaje_id, Order,Valor_accum,Flag], 
                arrayList=[NumeroTarjeta, TipoTarjeta, Valor, Fecha, Viaje_id, Order], 
                dtype=details_dtype,
                names=names
            )
#             print("details_arr")
#             print(details_arr)
        
#             print(current_process().name+" - "+str(a.dtype)+" - "+str(a.shape)+" - "+shm1_name)
            existing_shm1 = shared_memory.SharedMemory(name=shm1_name)
            tmp1 = np.ndarray(shape=details_arr.shape, dtype=details_arr.dtype, buffer=existing_shm1.buf)
            tmp1[:] = details_arr[:]
            
            print("Task "+str(id_batch)+" details processed in "+str((time.time() - start_time))+" seconds" )

            start_time = time.time()

            print("Task "+str(id_batch)+" Preparing for creating viajes...")

            viajes_dtype=[
                ('NumeroTarjeta', '<i8'),
                ('TipoTarjeta', '<i1'),
                ('dia', '<M8[D]'),
                ('Viaje_id', '<i8'),
                ('Order', '<i2'),
                ('Valor_Accum', '<i4'),
                ('Flag', '<i1')
            ]
#             print(Fecha)
#             print(Fecha.shape)
            # Convert YYYY-MM-DD HH:MM:SS to YYYY-MM-DD
            dia = Fecha//86400
            
#             print("Creating recarray...")
            names=('NumeroTarjeta', 'TipoTarjeta', 'dia', 'Viaje_id', 'Order', 'Valor_Accum','Flag')
            viajes_arr = np.rec.fromarrays(
                arrayList=[NumeroTarjeta, TipoTarjeta, dia, Viaje_id, Order, Valor_Accum, Flag], 
                dtype=viajes_dtype,
                names=names
            )
        
            # Leave only records with Flag == 1
            viajes_arr = viajes_arr[np.where(viajes_arr['Flag'] == 1)]

#             print("Selecting columns...")
            NumeroTarjeta = viajes_arr['NumeroTarjeta']
            TipoTarjeta   = viajes_arr['TipoTarjeta']
            dia           = viajes_arr['dia']
            Viaje_id      = viajes_arr['Viaje_id']
            Validaciones  = viajes_arr['Order']
            Valor_Accum   = viajes_arr['Valor_Accum']

            viajes_dtype=[
                ('NumeroTarjeta', '<i8'),
                ('TipoTarjeta', '<i1'),
                ('dia', '<M8[D]'),
                ('Viaje_id', '<i8'),
                ('Validaciones', '<i2'),
                ('Valor_Accum', '<i4'),
            ]
            names=('NumeroTarjeta', 'TipoTarjeta', 'dia', 'Viaje_id', 'Validaciones', 'Valor_Accum')
            
# #             print(viajes_arr)
# #             np_format = ['%i', '%i', '%i', '%i', '%i']
# #             header    = "NumeroTarjeta,TipoTarjeta,Viaje_id,Validaciones,Valor_Accum"
# #             np.savetxt("test.csv",np_arr,delimiter=",",fmt=np_format,header=header,comments="")
            
#             print("Creating recarray without Flag...")
            viajes_arr = np.rec.fromarrays(
                arrayList=[NumeroTarjeta, TipoTarjeta, dia, Viaje_id, Validaciones, Valor_Accum], 
                dtype=viajes_dtype,
                names=names
            )
        
#             print("Writing in shared memory...")
            existing_shm2 = shared_memory.SharedMemory(name=shm2_name)
            tmp2 = np.ndarray(shape=viajes_arr.shape, dtype=viajes_arr.dtype, buffer=existing_shm2.buf)
            tmp2[:] = viajes_arr[:]
            
            
            elem=(id_batch,shm1_name,details_arr.size,last_trip_id,shm2_name,viajes_arr.size)
            tasks_that_are_done.put(elem)
            print("task "+str(id_batch)+" viajes processed in "+str((time.time() - start_time))+" seconds" )

    return True

def uploading_np_arr(filetype,np_arr,np_format,month,gdrive_dir_id,header):
    start_time = time.time()
    filename=filetype+str(month)+".csv"
    print("Setting up gdrive command for uploading "+filename+"...")
    command=["/monitoreo/gdrive/gdrive","upload","-",filename,"-p", gdrive_dir_id]
    p = Popen(command, stdin=PIPE)#, stdout=PIPE, stderr=STDOUT)
    print("Uploading file: "+filename+"...")
    np.savetxt(p.stdin,np_arr,delimiter=",",fmt=np_format,header=header,comments="")
    print('File '+filename+' uploaded in %s seconds' % (time.time() - start_time))
    
def uploading_df(filetype,df,month,gdrive_dir_id):
    start_time = time.time()
    filename=filetype+str(month)+".csv"
    print("Setting up gdrive command for uploading "+filename+"...")
    command=["/monitoreo/gdrive/gdrive","upload","-",filename,"-p", gdrive_dir_id]
    p = Popen(command, stdin=PIPE)#, stdout=PIPE, stderr=STDOUT)
    print("Uploading file: "+filename+"...")
    df.to_csv(p.stdin,index=False)
    print('File '+filename+' uploaded in %s seconds' % (time.time() - start_time))
    
def saving_np_arr(filetype,np_arr,np_format,month,path,header):
    start_time = time.time()
    filename=filetype+str(month)+".csv"
    print("Saving file "+filename+"...")
    start_time = time.time()
    np.savetxt(path+filename,np_arr,delimiter=",",fmt=np_format,header=header,comments="")
    print('File '+filename+' saved in %s seconds' % (time.time() - start_time))
    
def saving_df(filetype,df,month,path):
    start_time = time.time()
    filename=filetype+str(month)+".csv"
    print("Saving file "+filename+"...")
    df.to_csv(filename,index=False)
    print('File '+filename+' saved in %s seconds' % (time.time() - start_time))
    
def process_period(num_processors,validations_batch_size,date_info,share_memory_bytes_details,share_memory_bytes_viajes,saving_files_flag,uploading_files_flag):
    #print(date_info)
    period_start_time = time.time()
    ini_date=date_info[0]
    end_date=date_info[1]
    month=date_info[2]
    max_validations=date_info[6]
    # Construct the query
    print("Getting validations count from BigQuery...")
    query = """
    SELECT distinct Numero_Tarjeta, count(Fecha_Clearing) as Num_val
    FROM `transmilenio-dwh-shvpc.validaciones.validacion` 
    WHERE (
        ( Fecha_Clearing between '"""+ini_date+"""' and DATE_ADD( DATE '"""+end_date+"""', INTERVAL 15 DAY) )
        and ( Fecha_Transaccion >= TIMESTAMP '"""+ini_date+""" 00:00:00') and (Fecha_Transaccion <= TIMESTAMP '"""+end_date+""" 23:59:59')
    ) 
    GROUP BY Numero_Tarjeta
    ORDER BY Numero_Tarjeta 
    """
    # Create a instance of bigquery 
    client = bigquery.Client()
    # API request
    df = client.query(query).to_dataframe(
        dtypes={"Numero_Tarjeta": "int64",
                "Num_val": "int32"})
    print("Validations count from BigQuery gotten!")

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
    with SharedMemoryManager() as smm:

        number_of_task = arr_len
        number_of_processes = num_processors
        tasks_to_accomplish = Queue()
        tasks_that_are_done = Queue()
        df_results          = Queue()
        processes = []

        for i in range(number_of_task):
            ini_card=card_num_array[i][0]
            end_card=card_num_array[i][1]
            #index=i+1
            shm1=smm.SharedMemory(size=share_memory_bytes_details)
            shm2=smm.SharedMemory(size=share_memory_bytes_viajes)
            elem=(i,number_of_task,ini_card,end_card,date_info,shm1.name,shm2.name)
            #print(elem)
            tasks_to_accomplish.put(elem)

        # creating processes
        print("Creating and starting workers...")
        for w in range(number_of_processes):
            p = Process(target=do_job_1, args=(tasks_to_accomplish, tasks_that_are_done))
            processes.append(p)
            p.start()
        print("Workers created and started!")

        # completing process
        print("Waiting for processes to finish...")
        for p in processes:
            p.join()
        print("Workers have finished!")


        print("Consolidating sub-results by month...")
        start_time = time.time()
        mtype=[
            ('id_batch', 'i1'), 
            ('shm_name1', 'U15'), 
            ('num_elems1','i8'), 
            ('last_trip_id','i8'),
            ('shm_name2','U15'),
            ('num_elems2','i8'),
        ]
        ordered_arr = np.zeros((0,), dtype=mtype)

        while not tasks_that_are_done.empty():
            task = tasks_that_are_done.get()
            ordered_arr = np.append(ordered_arr, np.array(task, dtype=mtype))

        # Order array by the first element (tup[0])
        ordered_arr = sorted(ordered_arr, key=lambda tup: tup[0])
#         print(ordered_arr)

        print("Consolidating arrays...")
        start_time = time.time() 
        details_dtype=[
            ('NumeroTarjeta', '<i8'),
            ('TipoTarjeta', '<i1'),
            ('Valor', '<i2'),
            ('Fecha', '<M8[s]'),
            ('Viaje_id', '<i8'),
            ('Order', '<i2')
        ]
        
        viajes_dtype=[
            ('NumeroTarjeta', '<i8'),
            ('TipoTarjeta', '<i1'),
            ('dia', '<M8[D]'),
            ('Viaje_id', '<i8'),
            ('Validaciones', '<i2'),
            ('Valor_Accum', '<i4'),
        ]
        details_arr = np.zeros(shape=(0,),dtype=details_dtype)
        viajes_arr = np.zeros(shape=(0,),dtype=viajes_dtype)
        offset_id_viaje=0
        for elem in ordered_arr:
            print(elem)
            id_batch     = elem[0]
            shm_name1    = elem[1]
            num_elems1   = elem[2]
            last_trip_id = elem[3]
            shm_name2    = elem[4]
            num_elems2   = elem[5]
#             print(id_batch)
#             print(str(id_batch)+" - "+str(num_elems)+" - "+str(last_trip_id)+" - "+shm_name+" - "+str(dtype))

            existing_shm1 = shared_memory.SharedMemory(name=shm_name1)
            existing_shm2 = shared_memory.SharedMemory(name=shm_name2)
            tmp_arr1 = np.recarray(shape=(num_elems1,), dtype=details_dtype, buf=existing_shm1.buf)
            tmp_arr2 = np.recarray(shape=(num_elems2,), dtype=viajes_dtype, buf=existing_shm2.buf)
            if id_batch>1:
                tmp_arr1['Viaje_id'] = tmp_arr1['Viaje_id'] + offset_id_viaje
#                 print(offset_id_viaje)
            details_arr = np.concatenate((details_arr, tmp_arr1), axis=0)
            viajes_arr = np.concatenate((viajes_arr, tmp_arr2), axis=0)
            offset_id_viaje = offset_id_viaje + last_trip_id
        
        print("Arrays consolidated in "+str((time.time() - start_time))+ " seconds")
        
        
    gdrive_dir_id="1YJo2TClzlXJVyPUv4z6eJYgoqTNI6ndl"
    path="/monitoreo/viajes_tmsa_andrea/"

    np_format1 = ['%i', '%i', '%i', '%s', '%i', '%i']
    np_format2 = ['%i', '%i', '%s', '%i', '%i', '%i']
    
    header1    = "NumeroTarjeta,TipoTarjeta,Valor,Fecha,Viaje_id,Orden"
    header2    = "NumeroTarjeta,TipoTarjeta,dia,Viaje_id,Validaciones,Valor_Accum"
    
    if(uploading_files_flag):
        uploading_details_proc = Process(target=uploading_np_arr, args=("detalles_",details_arr,np_format1,month,gdrive_dir_id,header1))
        uploading_details_proc.start()

        uploading_viajes_proc = Process(target=uploading_np_arr, args=("viajes_",viajes_arr,np_format2,month,gdrive_dir_id,header2))
        uploading_viajes_proc.start()
    
    if(saving_files_flag):
        saving_details_proc = Process(target=saving_np_arr, args=("detalles_",details_arr,np_format1,month,path,header1))
        saving_details_proc.start()

        saving_viajes_proc = Process(target=saving_np_arr, args=("viajes_",viajes_arr,np_format2,month,path,header2))
        saving_viajes_proc.start()

    start_time = time.time()
    print("Creating df_viajes...")
    columns=['NumeroTarjeta',
             'TipoTarjeta',
             'dia',
             'Viaje_id',
             'Validaciones',
             'Valor_Accum'
            ]

    df_viajes = pd.DataFrame(viajes_arr, columns=columns)
    dtype={'NumeroTarjeta': 'int64', 
           'TipoTarjeta': 'int8', 
           'dia': 'datetime64',
           'Viaje_id':'int64',
           'Validaciones':'int16',
           'Valor_Accum': 'int32'
          }
    df_viajes = df_viajes.astype(dtype)
    print("df_viajes created in "+str((time.time() - start_time))+ " seconds")
    
    start_time = time.time()
    print("Calculating info_tarjeta_dia...")
#     df_itd = df_viajes.loc[( (df_viajes['validaciones'] <= max_validations) & (df_viajes['mes'] == str(month)) )]

    df_itd = df_viajes.groupby(['NumeroTarjeta','dia']).agg(
        tipo_tarjeta=pd.NamedAgg(column='TipoTarjeta', aggfunc='last'),
        valor_dia=pd.NamedAgg(column='Valor_Accum', aggfunc=np.nansum),
        viajes_dia = pd.NamedAgg(column='Viaje_id', aggfunc='size'),
        validaciones_dia = pd.NamedAgg(column='Validaciones', aggfunc=np.nansum)).reset_index()

    if(uploading_files_flag):
        uploading_info_tarjeta_dia_proc = Process(target=uploading_df, args=("info_tarjeta_dia_",df_itd,month,gdrive_dir_id))
        uploading_info_tarjeta_dia_proc.start()

    if(saving_files_flag):
        saving_info_tarjeta_dia_proc = Process(target=saving_df, args=("info_tarjeta_dia_",df_itd,month,path))
        saving_info_tarjeta_dia_proc.start()

    print("info_tarjeta_dia calculated in "+str((time.time() - start_time))+ " seconds")

    # Add month to df_itd
    df_itd['mes'] = df_itd["dia"].dt.strftime("%Y-%m")
#     print(df_itd)
    start_time = time.time()
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

    if(uploading_files_flag):
        uploading_info_tarjeta_mes_proc = Process(target=uploading_df, args=("info_tarjeta_mes_",df_itm,month,gdrive_dir_id))
        uploading_info_tarjeta_mes_proc.start()
        
    if(saving_files_flag):
        saving_info_tarjeta_mes_proc = Process(target=saving_df, args=("info_tarjeta_mes_",df_itm,month,path))
        saving_info_tarjeta_mes_proc.start()
        
    print("info_tarjeta_mes calculated in "+str((time.time() - start_time))+ " seconds")

    print('Files Consolidated for month '+str(month)+'! --- %s seconds ---' % (time.time() - period_start_time))
    
if __name__ == '__main__':
    total_start_time = time.time()
    num_processors=multiprocessing.cpu_count()
    validations_batch_size     =  10000000
    share_memory_bytes_details = 300000000
    share_memory_bytes_viajes  = 300000000
    saving_files_flag    = True
    uploading_files_flag = False

    if (saving_files_flag):
        os.popen("rm *.csv")

    print("Getting trips info (number of validations for each card) ...")
    dates=[
        ['2019-09-01','2019-09-30','2019-09',1567295999,1569887999,95,3],
        ['2020-02-01','2020-02-29','2020-02',1580515199,1583020799,95,3],
        ['2020-10-01','2020-10-31','2020-10',1601510399,1604188799,110,4],
        
#         ['2019-09-29','2019-09-30','2019-09',1567295999,1569887999,95,3],
#         ['2020-02-01','2020-02-01','2020-02',1580515199,1583020799,95,3],
#         ['2020-10-01','2020-10-02','2020-10',1601510399,1604188799,110,4],
    ]
    
    for date_info in dates:
        date_start_time = time.time()
        print("Processing date: "+str(date_info[2]))
        process_period(num_processors,validations_batch_size,date_info,share_memory_bytes_details,share_memory_bytes_viajes,saving_files_flag,uploading_files_flag)
        print("--- Date " +date_info[2]+" Finished in %s seconds ---" % (time.time() - date_start_time))
    print("*** All Done! *** --- %s seconds ---" % (time.time() - total_start_time))
