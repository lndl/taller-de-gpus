#include "extras.h"

// Para medir los tiempos
double tick(){
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}

void array_init(double* a, int n, double k)
{
	int i;
	for(i=0; i<n; i++) {
		a[i] = k;
	}
}

void array_acc(double* acc, double* b, int n)
{
	int i;
	for(i=0; i<n; i++) {
		acc[i] += b[i];
	}
}

// Funcion para la inicializacion de las matrices
void init_matrix(data_t *M, const unsigned int size, int orientation, int k )
{
	unsigned int i,j;
	for (i=0; i<size; i++) {
		for (j=0; j<size; j++) {
			M[i*size + j] = 0;
			if ((orientation == 0) && (i==j)) 
				M[i*size + j] = k;
			if ((orientation == 1) && ((size-i-1) == j))
				M[i*size + j] = k;
		}
	}
}

// Verificador
float verificar(int n, int k1, int k2){
	return (n*(k1*k1+k2*k2))/(float)(n*n);
}

// Impresion de matriz (solo empleada en GPU para modo DEBUG)
void print_matrix(data_t * const M, const unsigned int size) {
  int i,j;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++)
      printf("%8d ", M[i*size+j]); 
    printf("\n");
  }
}

// Informe
void informar(double* times, double resultado, int N, int k1, int k2) {
	double testigo = verificar(N, k1, k2);
	
	printf("Alocar matrices en mem. de CPU:               %f\n", times[1]);
	printf("Inicializar matrices en mem. de CPU:          %f\n", times[2]);
	printf("Procesamiento en CPU sobre las matrices:      %f\n", times[3]);
	printf("Liberacion de  mem. CPU:                      %f\n", times[4]);
	printf("Tiempos para GPU:\n");
	printf("    Alocar matrices en mem. de GPU:           %f\n", times[6]);
	printf("    Tran. desde mem. CPU hacia mem. GPU:      %f\n", times[7]);
	printf("\x1B[32m    Ejecucion del kernel de GPU:              %f\n\x1B[0m", times[8]);
	printf("    Tran. desde mem. GPU hacia mem. CPU:      %f\n", times[9]);
	printf("Procesamiento total en la GPU:                %f\n", times[5]);
	printf("----------------------------------------------------------\n");
	printf("TIEMPO TOTAL:                                 %f\n", times[0]);
	printf("RESULTADO: %f = %f, %s\n", resultado, testigo, (resultado == testigo) ? "\x1B[32mCORRECTO\x1B[0m" : "\x1B[31mERRONEO\x1B[0m");
}
