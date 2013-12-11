#include <cuda.h>
#include <stdio.h>

#include "extras.h"
#include "gpu_func.h"

// Tipo de los datos del algoritmo
typedef int data_t;

// Prototipos kernels
__global__ void kernel_op_1(data_t * A, data_t * B);
__global__ void kernel_op_2(data_t * M, data_t* C, const unsigned int N);

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
  data_t *gpu_A, *gpu_B, *gpu_C;
  const unsigned int n_bytes = sizeof(data_t)*N*N;
  unsigned int x_bloques, y_bloques, n_threads;

  // Aloca memoria en GPU
  t[6] = tick();
  if (cudaSuccess != cudaMalloc((void**)&gpu_A, n_bytes))   printf("ERROR: INSUFICIENTE MEM EN LA GPU\n");
  if (cudaSuccess != cudaMalloc((void**)&gpu_B, n_bytes))   printf("ERROR: INSUFICIENTE MEM EN LA GPU\n");
  if (cudaSuccess != cudaMalloc((void**)&gpu_C, n_bytes/N)) printf("ERROR: INSUFICIENTE MEM EN LA GPU\n");
  t[6] = tick() - t[6];

  // Copia los datos desde el host a la GPU
  t[7] = tick();
  cudaMemcpy(gpu_A, host_A, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_B, host_B, n_bytes, cudaMemcpyHostToDevice);
  t[7] = tick() - t[7];

  // Configura el tamaño de los grids y los bloques
  n_threads = BLOCKS;
  calcular_dims(N, &x_bloques, &y_bloques, &n_threads, 1);
  dim3 dimGrid1(x_bloques, y_bloques);   
  dim3 dimBlock1(n_threads);
  calcular_dims(N, &x_bloques, &y_bloques, &n_threads, 0);
  dim3 dimGrid2(x_bloques, y_bloques);   
  dim3 dimBlock2(n_threads); 
  
  // Invoca al kernel
  t[8] = tick();
  kernel_op_1<<< dimGrid1, dimBlock1 >>>(gpu_A, gpu_B);
  cudaThreadSynchronize();
  //el segundo kernel usa shared mem del tamaño del bloque
  kernel_op_2<<< dimGrid2, dimBlock2, n_threads*sizeof(data_t) >>>(gpu_A, gpu_C, N);
  cudaThreadSynchronize();  
  t[8] = tick() - t[8];

  // Recupera los resultados, guardandolos en el host
  t[9] = tick();
  cudaMemcpy(host_A, gpu_A, n_bytes, cudaMemcpyDeviceToHost);
  data_t* host_C = (data_t*) malloc(n_bytes/N);
  cudaMemcpy(host_C, gpu_C, n_bytes/N, cudaMemcpyDeviceToHost);
  host_A[0] = 0;
  for(int i=0; i<N; i++) host_A[0] += host_C[i];
  free(host_C);
  t[9] = tick() - t[9];

  // Libera la memoria alocada en la GPU
  cudaFree(gpu_A);
  cudaFree(gpu_B);
  cudaFree(gpu_C);
}

// Los kernels que ejecutaran por cada hilo de la GPU
__global__ void kernel_op_1(data_t *A, data_t *B) {
  unsigned long int block_id =  blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long int global_id = block_id * blockDim.x + threadIdx.x;
  
  A[global_id] = (A[global_id] - B[global_id]) * (A[global_id] - B[global_id]);
}


__global__ void kernel_op_2(data_t *M, data_t *C, const unsigned int N) {
  unsigned long int block_id =  blockIdx.y * gridDim.x + blockIdx.x;
  unsigned long int global_id = block_id * blockDim.x + threadIdx.x;
  unsigned int i;
  extern __shared__ data_t S[]; //arreglo en shared mem
  S[threadIdx.x] = 0;
  for (i = 0; i < N; i++)
    S[threadIdx.x] += M[global_id + (N * i)];
  C[global_id] = S[threadIdx.x];
}
