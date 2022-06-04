#!/usr/bin/env python3
import time

import numpy as np
import psutil
from logger_tt import logger


def CHECKKEEPHEATINGSTATUS(temp_dict):
    smart_offset = 5
    temperatures = psutil.sensors_temperatures()
    # Save_Current_Temperatures
    memory_0400_temperature = temperatures[temp_dict['memory_temp_name']][0][1]
    memory_0100_temperature = temperatures[temp_dict['memory_temp_name']][1][1]
    keep_heating_memory = (memory_0400_temperature or memory_0100_temperature) \
                          < temp_dict['memory_critical']
    start_smart_memory_cooling = (memory_0400_temperature or memory_0100_temperature) \
                                 >= temp_dict['memory_critical'] + smart_offset
    stop_smart_memory_cooling = (memory_0400_temperature or memory_0100_temperature) \
                                <= temp_dict['memory_critical'] - smart_offset

    cpu_1_temperature = temperatures[temp_dict['cpu_temp_name']][0][1]
    cpu_2_temperature = temperatures[temp_dict['cpu_temp_name']][1][1]
    keep_heating_cpu = (cpu_1_temperature or cpu_2_temperature) \
                       < temp_dict['cpu_critical']
    start_smart_cpu_cooling = (cpu_1_temperature or cpu_2_temperature) \
                              >= temp_dict['cpu_critical'] + smart_offset
    stop_smart_cpu_cooling = (cpu_1_temperature or cpu_2_temperature) \
                             <= temp_dict['cpu_critical'] - smart_offset

    # gpu_edge_temperature = temperatures[temp_dict['gpu_temp_name']][0][1]
    # gpu_junction_temperature = temperatures[temp_dict['gpu_temp_name']][1][1]
    # gpu_mem_temperature = temperatures[temp_dict['gpu_temp_name']][2][1]
    # keep_heating_gpu = (gpu_edge_temperature or gpu_junction_temperature or gpu_mem_temperature) \
    #                    < temp_dict['gpu_critical']
    # start_smart_gpu_cooling = (gpu_edge_temperature or gpu_junction_temperature or gpu_mem_temperature) \
    #                           >= temp_dict['gpu_critical'] + smart_offset
    # stop_smart_gpu_cooling = (gpu_edge_temperature or gpu_junction_temperature or gpu_mem_temperature) \
    #                          <= temp_dict['gpu_critical'] - smart_offset

    Temp_Results = dict(
        memory_0400_temperature=memory_0400_temperature,
        memory_0100_temperature=memory_0100_temperature,

        cpu_1_temperature=cpu_1_temperature,
        cpu_2_temperature=cpu_2_temperature,

        # gpu_edge_temperature=gpu_edge_temperature,
        # gpu_junction_temperature=gpu_junction_temperature,
        # gpu_mem_temperature=gpu_mem_temperature,

        # keep_heating=np.array([keep_heating_memory, keep_heating_cpu, keep_heating_gpu]),
        # start_smart_cooling=np.array([start_smart_memory_cooling, start_smart_cpu_cooling, start_smart_gpu_cooling]),
        # stop_smart_cooling=np.array([stop_smart_memory_cooling, stop_smart_cpu_cooling, stop_smart_gpu_cooling])
        keep_heating=np.array([keep_heating_memory, keep_heating_cpu]),
        start_smart_cooling=np.array([start_smart_memory_cooling, start_smart_cpu_cooling]),
        stop_smart_cooling=np.array([stop_smart_memory_cooling, stop_smart_cpu_cooling])

    )
    return Temp_Results


def CHECKTEMPS(temp_dict, verbose=False):
    START_SMART_COOLING = False
    temperature_results = CHECKKEEPHEATINGSTATUS(temp_dict)
    # if not all(temperature_results['keep_heating']):
    #     logger.warning('Cooling, let me catch my breath!')
    while not all(temperature_results['keep_heating']) or START_SMART_COOLING:
        if verbose:
            for i in temperature_results:
                logger.warning(f'{i} = {temperature_results[i]}')
        time.sleep(temp_dict['Time_Dalay'])
        temperature_results = CHECKKEEPHEATINGSTATUS(temp_dict)
        if any(temperature_results['start_smart_cooling']) or START_SMART_COOLING:
            if not all(temperature_results['stop_smart_cooling']):
                if not START_SMART_COOLING:
                    logger.warning('Smart Cooling: On            Please Wait :)')
                    START_SMART_COOLING = True
            else:
                START_SMART_COOLING = False
                logger.warning('Smart Cooling: off')
