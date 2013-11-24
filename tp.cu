#include <cuda.h>
#include <stdio.h>

#define N 16

// Tipo de los datos del algoritmo
typedef int data_t;

// Prototipos 
data_t  add(const data_t a, const data_t b) { return a + b; }
data_t  sub(const data_t a, const data_t b) { return a - b; }
void    init_matrix(data_t *M, const unsigned int size, data_t(*init_op)(const data_t, const data_t));
void    run_GPU(data_t* host_A, data_t* host_B, const unsigned int n_bytes);

__global__ void kernel_operations(data_t * A, data_t * B);

// Host function
int
main(int argc, char** argv)
{
  //const int N = (argc == 2) ? atoi(argv[1]) : 0;
  
  if (!N){
    printf("Parametros incorrectos. El programa se cierra\n");
    return -1;
  } 

  // En la CPU...
  // ...Aloca matrices
  const unsigned int n_bytes = sizeof(data_t)*N*N;
  data_t *host_A = (data_t*) malloc(n_bytes);
  data_t *host_B = (data_t*) malloc(n_bytes);

  // ...Inicializa matrices
  init_matrix(host_A, N, &add);
  init_matrix(host_B, N, &sub);

  run_GPU(host_A, host_B, n_bytes);

  free(host_A);
  free(host_B);

  return 0;
}

void 
run_GPU(data_t* host_A, data_t* host_B, const unsigned int n_bytes)
{
  data_t *gpu_A, *gpu_B;
  int i, j;
  // Aloca memoria en GPU
  cudaMalloc((void**)&gpu_A, n_bytes);
  cudaMalloc((void**)&gpu_B, n_bytes);

  host_A[0] = 1;
  // Copia los datos desde el host a la GPU
  cudaMemcpy(gpu_A, host_A, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_B, host_B, n_bytes, cudaMemcpyHostToDevice);

  // Configura el tamaño de los grids y los bloques
  dim3 dimGrid(16);   // one block per word  
  dim3 dimBlock(16); // one thread per character
  
  // Invoca al kernel
  kernel_operations<<< dimGrid, dimBlock >>>(gpu_A, gpu_B);

  // Recupera los resultados, guardandolos en el host
  cudaMemcpy(host_A, gpu_A, n_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_B, gpu_B, n_bytes, cudaMemcpyDeviceToHost);
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			printf("%8d ",host_A[i*N+j]);
		}
	printf("\n");
	}
  // Libera la memoria alocada en la GPU
  cudaFree(gpu_A);
  cudaFree(gpu_B);
}

// El kernel que ejecutara en cada hilo de la GPU
__global__ void kernel_operations(data_t *A, data_t *B){
	unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	A[global_id] = (A[global_id]-B[global_id])*(A[global_id]-B[global_id]);
	//A[global_id] = 2;
}

// Funcion para la inicializacion de las matrices
void 
init_matrix(data_t *M, const unsigned int size, data_t(*init_op)(const data_t, const data_t))
{
  unsigned int i,j;
  for (i=0; i<size; i++) {
    for (j=0; j<size; j++) {
      M[i*size + j] = (*init_op)(i,j);
    }
  }
}
