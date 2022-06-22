#!/usr/bin/env python3
TEMP_DICT = dict(
    Time_Dalay=0.1,  # in Seconds to cool

    memory_temp_name='nvme',
    memory_critical=65,  # 74.8
    # array_memory_0400_temperature=np.array([]),
    # array_memory_0100_temperature=np.array([]),

    cpu_temp_name='k10temp',
    cpu_critical=80,  # 80
    # array_cpu_1_temperature=np.array([]),
    # array_cpu_2_temperature=np.array([]),

    gpu_temp_name='amdgpu',
    gpu_critical=80,  # 100
    # array_gpu_edge_temperature=np.array([]),
    # array_gpu_junction_temperature=np.array([]),
    # array_gpu_mem_temperature=np.array([]),
)
