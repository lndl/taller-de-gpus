#include "gpu_func.h"
#include "extras.h"

void gpu_init_and_shutdown(int argc, char** argv)
{
  const unsigned int n      = (argc >= 2) ? atoi(argv[1]) : 8;
  const unsigned int blocks = (argc >= 3) ? atoi(argv[2]) : 64;
  const unsigned int count  = (argc >= 4) ? atoi(argv[3]) : 1;
  const unsigned int k1     = (argc >= 5) ? atoi(argv[4]) : 7;
  const unsigned int k2     = (argc >= 6) ? atoi(argv[5]) : 9;
  unsigned int i;
  double t_acc[10], resultado;

  array_init(t_acc, 10, 0);
  for(i=0; i<count; i++)
  {
    double t[10];
    array_init(t, 10, -1);
    resultado = gpu_exec(n,blocks,k1,k2,t);
    array_acc(t_acc, t, 10);
  }
  informar(t_acc, resultado, n, k1, k2);
}

// Arrancador (solo GPU)
double gpu_exec(unsigned int n, unsigned int blocks, unsigned int k1, unsigned int k2, double t[])
{
  double resultado;  

  #ifdef DEBUG
  // Mostrar tipo de elemento
  printf("Tamaño del elemento a procesar: %d bytes\n", sizeof(data_t));
  #endif

  t[0] = tick(); 

  // En la CPU...
  // ...Aloca matrices
  t[1] = tick();
  const unsigned int n_bytes = sizeof(data_t)*n*n;
  data_t *host_A = (data_t*) malloc(n_bytes);
  data_t *host_B = (data_t*) malloc(n_bytes);
  t[1] = tick() - t[1];

  // ...Inicializa matrices
  t[2] = tick();
  init_matrix(host_A, n, 0, k1);
  init_matrix(host_B, n, 1, k2);
  t[2] = tick() - t[2];

  #ifdef DEBUG
  printf("Matriz A =====\n");
  print_matrix(host_A, n);
  printf("Matriz B =====\n");
  print_matrix(host_B, n);
  #endif

  // ...a procesar a la GPU
  t[5] = tick();
  run_GPU(host_A, host_B, n, blocks, t);
  t[5] = tick() - t[5];

  #ifdef DEBUG
  printf("Dump de matriz resultado =====\n");
  print_matrix(host_A, n);
  #endif

  // Paso final: dividir la suma
  resultado = host_A[0]/((double)n*n);

  t[4] = tick();
  free(host_A);
  free(host_B);
  t[4] = tick() - t[4];

  t[0] = tick() - t[0];

  return resultado;
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
}
