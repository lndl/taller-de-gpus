#!/bin/bash

echo "COMPILANDO"
echo "gcc tp-secuencial.c extras.c -o secuencial"
gcc tp-secuencial.c extras.c -o secuencial
echo "nvcc tp-no-optimizado.cu gpu_func.c extras.c -o no-optimizado"
nvcc tp-no-optimizado.cu gpu_func.c extras.c -o no-optimizado
echo "nvcc tp-optimizado.cu gpu_func.c extras.c -o optimizado"
nvcc tp-optimizado.cu gpu_func.c extras.c -o optimizado
echo "nvcc tp-optimizado-shared.cu gpu_func.c extras.c -o optimizado-shared"
nvcc tp-optimizado-shared.cu gpu_func.c extras.c -o optimizado-shared
