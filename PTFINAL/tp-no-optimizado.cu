#include <cuda.h>
#include <stdio.h>

#include "extras.h"
#include "gpu_func.h"

// Tipo de los datos del algoritmo
typedef int data_t;

// Prototipos kernels
__global__ void kernel_op_1(data_t * A, data_t * B);
__global__ void kernel_op_2(data_t * M, const unsigned int size, const unsigned int limit);

// Host function
int
main(int argc, char** argv)
{
  gpu_init_and_shutdown(argc, argv);
  return 0;
}

void 
run_GPU(data_t* host_A, data_t* host_B, const unsigned int N, const unsigned int BLOCKS, double* t)
{
  data_t *gpu_A, *gpu_B;
  const unsigned int n_bytes = sizeof(data_t)*N*N;
  unsigned int i, x_bloques, y_bloques, n_threads;

  // Aloca memoria en GPU
  t[6] = tick();
  cudaMalloc((void**)&gpu_A, n_bytes);
  cudaMalloc((void**)&gpu_B, n_bytes);
  t[6] = tick() - t[6];
  

  // Copia los datos desde el host a la GPU
  t[7] = tick();
  cudaMemcpy(gpu_A, host_A, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_B, host_B, n_bytes, cudaMemcpyHostToDevice);
  t[7] = tick() - t[7];
  

  // Configura el tamaño de los grids y los bloques
  n_threads = BLOCKS;
  calcular_dims(N, &x_bloques, &y_bloques, &n_threads, 1);
  dim3 dimGrid(x_bloques, y_bloques);   
  dim3 dimBlock(n_threads); 
  
  // Invoca al kernel
  t[8] = tick();
  kernel_op_1<<< dimGrid, dimBlock >>>(gpu_A, gpu_B);
  cudaThreadSynchronize();
  for (i=1; i<N*N; i*=2) {
    kernel_op_2<<< dimGrid, dimBlock >>>(gpu_A, i, N*N);
    cudaThreadSynchronize();
  }
  t[8] = tick() - t[8];

  // Recupera los resultados, guardandolos en el host
  t[9] = tick();
  cudaMemcpy(host_A, gpu_A, n_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_B, gpu_B, n_bytes, cudaMemcpyDeviceToHost);
  t[9] = tick() - t[9];

  // Libera la memoria alocada en la GPU
  cudaFree(gpu_A);
  cudaFree(gpu_B);
}

// Los kernels que ejecutaran por cada hilo de la GPU
__global__ void kernel_op_1(data_t *A, data_t *B) {
  unsigned long int block_id =  blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long int global_id = block_id * blockDim.x + threadIdx.x;

  A[global_id] = (A[global_id] - B[global_id]) * (A[global_id] - B[global_id]);
}

__global__ void kernel_op_2(data_t *M, const unsigned int offset, const unsigned int limit) {
  unsigned long int block_id =  blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long int global_id = block_id * blockDim.x + threadIdx.x;

  if (global_id + offset <= limit)
    M[global_id] += M[global_id + offset]; 
}
