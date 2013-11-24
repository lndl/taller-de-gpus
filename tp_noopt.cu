#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>

// Tipo de los datos del algoritmo
typedef int data_t;

// Prototipos 
data_t  add(const data_t a, const data_t b) { return a + b; }
data_t  sub(const data_t a, const data_t b) { return a - b; }
void    init_matrix(data_t *M, const unsigned int size, data_t(*init_op)(const data_t, const data_t));
void    run_GPU(data_t* host_A, data_t* host_B, const unsigned int n_bytes);
void    print_matrix(data_t * const M, const unsigned int size);
double tick();

__global__ void kernel_op_1(data_t * A, data_t * B);
__global__ void kernel_op_2(data_t * M, const unsigned int size);

// Host function
int
main(int argc, char** argv)
{
  const unsigned int N = (argc == 2) ? atoi(argv[1]) : 0;
  double t, resultado;
  
  if (!N){
    printf("Parametros incorrectos. El programa se cierra\n");
    return -1;
  } 

  // Mostrar tipo de elemento
  printf("Tamaño del elemento a procesar: %d bytes\n", sizeof(data_t));

  // En la CPU...
  // ...Aloca matrices
  t = tick();
  const unsigned int n_bytes = sizeof(data_t)*N*N;
  data_t *host_A = (data_t*) malloc(n_bytes);
  data_t *host_B = (data_t*) malloc(n_bytes);
  t = tick() - t;
  printf("Alocar matrices en mem. de CPU: %f\n", t);

  // ...Inicializa matrices
  t = tick();
  init_matrix(host_A, N, &add);
  init_matrix(host_B, N, &sub);
  t = tick() - t;
  printf("Inicializar matrices en mem. de CPU: %f\n", t);

  #ifdef DEBUG
  printf("Matriz A =====\n");
  print_matrix(host_A, N);
  printf("Matriz B =====\n");
  print_matrix(host_B, N);
  #endif

  run_GPU(host_A, host_B, n_bytes);

  // Verificacion de resultados
  #ifdef DEBUG
  printf("Resultado parcial =====\n");
  print_matrix(host_A, N);
  #endif

  //Paso final: dividir la suma
  resultado = host_A[0]/((float)N*N);

  t = tick();
  free(host_A);
  free(host_B);
  t = tick() - t;
  printf("Liberacion de  mem. CPU: %f\n", t);

  printf("\x1B[36mResultado final  =====>>>  %f\x1B[0m\n", resultado);

  return 0;
}

void 
run_GPU(data_t* host_A, data_t* host_B, const unsigned int n_bytes)
{
  data_t *gpu_A, *gpu_B;
  const unsigned int size = n_bytes / sizeof(data_t);
  unsigned int i;
  double t;

  // Aloca memoria en GPU
  t = tick();
  cudaMalloc((void**)&gpu_A, n_bytes);
  cudaMalloc((void**)&gpu_B, n_bytes);
  t = tick() - t;
  printf("Alocar matrices en mem. de GPU: %f\n", t);

  // Copia los datos desde el host a la GPU
  t = tick();
  cudaMemcpy(gpu_A, host_A, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_B, host_B, n_bytes, cudaMemcpyHostToDevice);
  t = tick() - t;
  printf("Copia de datos desde mem. CPU hacia mem. GPU: %f\n", t);

  // Configura el tamaño de los grids y los bloques
  dim3 dimGrid(1);   
  dim3 dimBlock(16); 
  
  // Invoca al kernel
  t = tick();
  kernel_op_1<<< dimGrid, dimBlock >>>(gpu_A, gpu_B);
  cudaThreadSynchronize();
  for (i=1; i<size; i*=2) {
    kernel_op_2<<< dimGrid, dimBlock >>>(gpu_A, i);
    cudaThreadSynchronize();
  }
  t = tick() - t;
  printf("\x1B[33mEjecucion del kernel de GPU: %f\x1B[0m\n", t);

  // Recupera los resultados, guardandolos en el host
  t = tick();
  cudaMemcpy(host_A, gpu_A, n_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_B, gpu_B, n_bytes, cudaMemcpyDeviceToHost);
  t = tick() - t;
  printf("Copia de datos desde mem. GPU hacia mem. CPU: %f\n", t);

  // Libera la memoria alocada en la GPU
  t = tick();
  cudaFree(gpu_A);
  cudaFree(gpu_B);
  t = tick() - t;
  printf("Liberar mem. de GPU: %f\n", t);
}

// Los kernels que ejecutaran por cada hilo de la GPU
__global__ void kernel_op_1(data_t *A, data_t *B) {
  unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  A[global_id] = (A[global_id] - B[global_id]) * (A[global_id] - B[global_id]);
}

__global__ void kernel_op_2(data_t *M, const unsigned int offset) {
  unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  M[global_id] += M[global_id + offset]; 
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

// Impresion de matriz
void print_matrix(data_t * const M, const unsigned int size) {
  int i,j;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++)
        printf("%8d ", M[i*size+j]); 
    printf("\n");
  }
}

// Para medir los tiempos
double tick(){
  double sec;
  struct timeval tv;

  gettimeofday(&tv,NULL);
  sec = tv.tv_sec + tv.tv_usec/1000000.0;
  return sec;
}
