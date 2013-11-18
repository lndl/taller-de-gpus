#include <cuda.h>
#include <stdio.h>

// Tipo de los datos del algoritmo
typedef double data_t;

// Prototipos 
data_t  add(const data_t a, const data_t b) { return a + b; }
data_t  sub(const data_t a, const data_t b) { return a - b; }
void    init_matrix(data_t *M, const unsigned int size, data_t(*init_op)(const data_t, const data_t));
void    run_GPU(data_t* host_A, data_t* host_B, const unsigned int n_bytes);

__global__ void kernel_operations();

// Host function
int
main(int argc, char** argv)
{
  const int N = (argc == 2) ? atoi(argv[1]) : 0;
  
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

  // Aloca memoria en GPU
  cudaMalloc((void**)&gpu_A, n_bytes);
  cudaMalloc((void**)&gpu_B, n_bytes);

  // Copia los datos desde el host a la GPU
  cudaMemcpy(gpu_A, host_A, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_B, host_B, n_bytes, cudaMemcpyHostToDevice);

  // Configura el tamaño de los grids y los bloques
  dim3 dimGrid(2);   // one block per word  
  dim3 dimBlock(6); // one thread per character
  
  // Invoca al kernel
  kernel_operations<<< dimGrid, dimBlock >>>();

  // Recupera los resultados, guardandolos en el host
  cudaMemcpy(host_A, gpu_A, n_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_B, gpu_B, n_bytes, cudaMemcpyDeviceToHost);

  // Libera la memoria alocada en la GPU
  cudaFree(gpu_A);
  cudaFree(gpu_B);
}

// El kernel que ejecutara en cada hilo de la GPU
__global__ void
kernel_operations()
{
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
