#ifndef GPU_FUNC_H
#define GPU_FUNC_H

#include "extras.h"

void gpu_init_and_shutdown(int argc, char** argv);
/* La implementacion en funcion de los distintos programas */
void run_GPU(data_t* host_A, data_t* host_B, const unsigned int n_bytes, const unsigned int BLOCKS, double* times);
void calcular_dims(const unsigned int n, unsigned int* x_bloques,unsigned int* y_bloques, unsigned int* n_threads, int ismatrix); 

#endif
