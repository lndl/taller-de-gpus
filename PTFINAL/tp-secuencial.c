#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef int data_t;
// Para medir los tiempos

double tick(){
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}

// Funcion para la inicializacion de las matrices
void init_matrix(data_t *M, const unsigned int size, int orientation, int k )
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

//  Función secuencial: MSE
double MSE(int *a, int *b, const unsigned int n) {
	unsigned int i;
	double total=0;
	for(i = 0; i < n*n; i++) {
		total += ((a[i] - b[i])*((a[i] - b[i])));
	}
	total/=n*n;
	return total;
}

float verificar(int n, int k1, int k2){
	return (n*(k1*k1+k2*k2))/(float)(n*n);
}

int main(int argc, char *argv[]){
	int * a, * b;
	double t, resultado, testigo;
	const unsigned int N  = (argc >= 2) ? atoi(argv[1]) : 8;
	const unsigned int k1 = (argc >= 3) ? atoi(argv[2]) : 7;
	const unsigned int k2 = (argc >= 4) ? atoi(argv[3]) : 9;
	
	// Mostrar tipo de elemento
  printf("Tamaño del elemento a procesar: %d bytes\n", sizeof(data_t));
	
	t = tick();
	a = malloc(N*N*sizeof(data_t));
	b = malloc(N*N*sizeof(data_t));
	t = tick() - t;
	printf("Alocar matrices en mem. de CPU: %f\n", t);

	t = tick();
	init_matrix(a, N, 0, k1);
	init_matrix(b, N, 1, k2);
	t = tick() - t;
	printf("Inicializar matrices en mem. de CPU: %f\n", t);
	
	t = tick();
	resultado = MSE(a,b,N);
	t = tick() - t;
	printf("Procesamiento en CPU sobre las matrices: %f\n", t);
	
	t = tick();
	free(a);
	free(b);
	t = tick() - t;
	printf("Liberacion de  mem. CPU: %f\n", t);
	
	printf("\x1B[34mResultado final >>> %f\x1B[0m\n", resultado);
	if (resultado == (testigo = verificar (N, k1, k2)))
		printf("\x1B[32mVerificación >>> Valor esperado: %f | Valor obtenido: %f\x1B[0m\n", testigo, resultado);
	else
		printf("\x1B[31mVerificación >>> Valor esperado: %f | Valor obtenido: %f\x1B[0m\n", testigo, resultado);
	return 0;
}
