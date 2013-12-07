#include "gpu_func.h"
#include "extras.h"

// Arrancador (solo GPU)
void gpu_init_and_shutdown(int argc, char** argv)
{
  const unsigned int N      = (argc >= 2) ? atoi(argv[1]) : 8;
  const unsigned int BLOCKS = (argc >= 3) ? atoi(argv[2]) : 64;
  const unsigned int k1     = (argc >= 4) ? atoi(argv[3]) : 7;
  const unsigned int k2     = (argc >= 5) ? atoi(argv[4]) : 9;
  double t[10], resultado;
  
  #ifdef DEBUG
  // Mostrar tipo de elemento
  printf("Tamaño del elemento a procesar: %d bytes\n", sizeof(data_t));
  #endif

  t[0] = tick(); 

  // En la CPU...
  // ...Aloca matrices
  t[1] = tick();
  const unsigned int n_bytes = sizeof(data_t)*N*N;
  data_t *host_A = (data_t*) malloc(n_bytes);
  data_t *host_B = (data_t*) malloc(n_bytes);
  t[1] = tick() - t[1];

  // ...Inicializa matrices
  t[2] = tick();
  init_matrix(host_A, N, 0, k1);
  init_matrix(host_B, N, 1, k2);
  t[2] = tick() - t[2];

  #ifdef DEBUG
  printf("Matriz A =====\n");
  print_matrix(host_A, N);
  printf("Matriz B =====\n");
  print_matrix(host_B, N);
  #endif

  // ...a procesar a la GPU
  t[5] = tick();
  run_GPU(host_A, host_B, N, BLOCKS, t);
  t[5] = tick() - t[5];

  #ifdef DEBUG
  printf("Dump de matriz resultado =====\n");
  print_matrix(host_A, N);
  #endif

  // Paso final: dividir la suma
  resultado = host_A[0]/((double)N*N);

  t[4] = tick();
  free(host_A);
  free(host_B);
  t[4] = tick() - t[4];
  printf("Liberacion de  mem. CPU: %f\n", t);

  t[0] = tick() - t[0];

  informar(t, resultado, N, k1, k2);
}


// Balanceador (solo GPU)
void calcular_dims(const unsigned int n, unsigned int* x_bloques,unsigned int* y_bloques, unsigned int* n_threads, int ismatrix) {
  int N = (ismatrix) ? n*n : n ;
  *x_bloques = ((N)/(*n_threads));
  if (*x_bloques == 0) {
    *x_bloques = 1;
    printf("    \x1B[33mNOTICE: Reajustando x_bloques -> %d\x1B[0m\n", *x_bloques);
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
		printf("    \x1B[33mNOTICE: Reajustando x_bloques -> %d\x1B[0m\n", *x_bloques);
		printf("    \x1B[33mNOTICE: Reajustando y_bloques -> %d\x1B[0m\n", *y_bloques);
  }
  if (*x_bloques > 65535) {
    printf("    \x1B[31mWARNING: Número de BLOQUES!! mayor al soportado por la placa!!\x1B[0m\n");
  }
  //printf("    \x1B[31mWARNING: Número de BLOQUESXX!! %d!!\x1B[0m\n", *x_bloques);
  //printf("    \x1B[31mWARNING: Número de BLOQUESYY!! %d!!\x1B[0m\n", *y_bloques);
}
