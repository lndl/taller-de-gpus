#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>

// Tipo de los datos del algoritmo
typedef int data_t;

// Prototipos
void    init_matrix(data_t *M, const unsigned int size, int orientation, int k);
void    run_GPU(data_t* host_A, data_t* host_B, const unsigned int n_bytes, const unsigned int BLOCKS);
void    print_matrix(data_t * const M, const unsigned int size);
double  tick();
float   verificar(int n, int k1, int k2);
void    calcular_dims(const unsigned int n, unsigned int* x_bloques, unsigned int* y_bloques, unsigned int* n_threads, int ismatrix); 

__global__ void kernel_op_1(data_t * A, data_t * B);
__global__ void kernel_op_2(data_t * M, const unsigned int size, const unsigned int limit);

// Host function
int
main(int argc, char** argv)
{
  const unsigned int N  = (argc >= 2) ? atoi(argv[1]) : 8;
  const unsigned int BLOCKS = (argc >= 3) ? atoi(argv[2]) : 64;
  const unsigned int k1 = (argc >= 4) ? atoi(argv[3]) : 7;
  const unsigned int k2 = (argc >= 5) ? atoi(argv[4]) : 9;
  double t, resultado, testigo;
  
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
  init_matrix(host_A, N, 0, k1);
  init_matrix(host_B, N, 1, k2);
  t = tick() - t;
  printf("Inicializar matrices en mem. de CPU: %f\n", t);

  #ifdef DEBUG
  printf("Matriz A =====\n");
  print_matrix(host_A, N);
  printf("Matriz B =====\n");
  print_matrix(host_B, N);
  #endif

  // ...a procesar a la GPU
  t = tick();
  printf("-- Procesamiento en GPU sobre las matrices:\n");
  run_GPU(host_A, host_B, N, BLOCKS);
  t = tick() - t;
  printf("-- Procesamiento total en la GPU: %f\n", t);

  #ifdef DEBUG
  printf("Dump de matriz resultado =====\n");
  print_matrix(host_A, N);
  #endif

  // Paso final: dividir la suma
  resultado = host_A[0]/((double)N*N);

  t = tick();
  free(host_A);
  free(host_B);
  t = tick() - t;
  printf("Liberacion de  mem. CPU: %f\n", t);

  // Verificacion de resultados
  printf("\x1B[33mResultado final >>> %f\x1B[0m\n", resultado);
  if (resultado == (testigo = verificar (N, k1, k2)))
    printf("\x1B[32mVerificación >>> Valor esperado: %f | Valor obtenido: %f\x1B[0m\n", testigo, resultado);
  else
    printf("\x1B[31mVerificación >>> Valor esperado: %f | Valor obtenido: %f\x1B[0m\n", testigo, resultado);
  return 0;
}

void 
run_GPU(data_t* host_A, data_t* host_B, const unsigned int N, const unsigned int BLOCKS)
{
  data_t *gpu_A, *gpu_B;
  const unsigned int n_bytes = sizeof(data_t)*N*N;
  unsigned int i, x_bloques, y_bloques, n_threads;
  double t;

  // Aloca memoria en GPU
  t = tick();
  cudaMalloc((void**)&gpu_A, n_bytes);
  cudaMalloc((void**)&gpu_B, n_bytes);
  t = tick() - t;
  printf("    Alocar matrices en mem. de GPU: %f\n", t);

  // Copia los datos desde el host a la GPU
  t = tick();
  cudaMemcpy(gpu_A, host_A, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_B, host_B, n_bytes, cudaMemcpyHostToDevice);
  t = tick() - t;
  printf("    Copia de datos desde mem. CPU hacia mem. GPU: %f\n", t);

  // Configura el tamaño de los grids y los bloques
  n_threads = BLOCKS;
  calcular_dims(N, &x_bloques, &y_bloques, &n_threads, 1);
  dim3 dimGrid(x_bloques, y_bloques);   
  dim3 dimBlock(n_threads); 
  
  // Invoca al kernel
  t = tick();
  kernel_op_1<<< dimGrid, dimBlock >>>(gpu_A, gpu_B);
  cudaThreadSynchronize();
  for (i=1; i<N*N; i*=2) {
    kernel_op_2<<< dimGrid, dimBlock >>>(gpu_A, i, N*N);
    cudaThreadSynchronize();
  }
  t = tick() - t;
  printf("    \x1B[33mEjecucion del kernel de GPU: %f\x1B[0m\n", t);

  // Recupera los resultados, guardandolos en el host
  t = tick();
  cudaMemcpy(host_A, gpu_A, n_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_B, gpu_B, n_bytes, cudaMemcpyDeviceToHost);
  t = tick() - t;
  printf("    Copia de datos desde mem. GPU hacia mem. CPU: %f\n", t);

  // Libera la memoria alocada en la GPU
  t = tick();
  cudaFree(gpu_A);
  cudaFree(gpu_B);
  t = tick() - t;
  printf("    Liberar mem. de GPU: %f\n", t);
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

// Funcion para la inicializacion de las matrices
void 
init_matrix(data_t *M, const unsigned int size, int orientation, int k )
{
  unsigned int i,j;
  for (i=0; i<size; i++) {
    for (j=0; j<size; j++) {
      if ((orientation == 0) && (i==j)){
        M[i*size + j] = k;
      }
      if ((orientation == 1) && ((size-i-1) == j)){
        M[i*size + j] = k;
      }
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

float verificar(int n, int k1, int k2){
  return (n*(k1*k1+k2*k2))/(float)(n*n);
}

void calcular_dims(const unsigned int n, unsigned int* x_bloques,unsigned int* y_bloques, unsigned int* n_threads, int ismatrix) {
  int N = (ismatrix) ? n*n : n ;
  *x_bloques = ((N)/(*n_threads));
  if (*x_bloques == 0) {
    *x_bloques = 1;
  }
  if (*n_threads > 1024) {
    printf("    \x1B[31mWARNING: Número de threads mayor al soportado por la placa!!\x1B[0m\n");
  }
  *y_bloques = 1;
  if (*x_bloques > 65535) {
    double n = *x_bloques / 65535.0;
    unsigned int i;
    for (i = 1; i < n; i *= 2);
    *y_bloques = i;
    *x_bloques /= *y_bloques;
  }
  if (*x_bloques > 65535) {
    printf("    \x1B[31mWARNING: Número de BLOQUES!! mayor al soportado por la placa!!\x1B[0m\n");
  }
  //printf("    \x1B[31mWARNING: Número de BLOQUESXX!! %d!!\x1B[0m\n", *x_bloques);
  //printf("    \x1B[31mWARNING: Número de BLOQUESYY!! %d!!\x1B[0m\n", *y_bloques);
}

