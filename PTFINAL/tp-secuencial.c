#include <stdio.h>
#include <stdlib.h>

#include "extras.h"

//  Función secuencial: MSE
double MSE(data_t *a, data_t *b, const unsigned int n) {
	unsigned int i;
	double total=0;
	for(i = 0; i < n*n; i++) {
		total += ((a[i] - b[i])*((a[i] - b[i])));
	}
	total/=n*n;
	return total;
}

int main(int argc, char *argv[]){
	int * a, * b;
	double t[10], resultado;
	const unsigned int N  = (argc >= 2) ? atoi(argv[1]) : 8;
	const unsigned int k1 = (argc >= 3) ? atoi(argv[2]) : 7;
	const unsigned int k2 = (argc >= 4) ? atoi(argv[3]) : 9;

	t[0] = tick();
	
	t[1] = tick();
	a = malloc(N*N*sizeof(data_t));
	b = malloc(N*N*sizeof(data_t));
	t[1] = tick() - t[1];
	
	t[2] = tick();
	init_matrix(a, N, 0, k1);
	init_matrix(b, N, 1, k2);
	t[2] = tick() - t[2];
	
	
	t[3] = tick();
	resultado = MSE(a,b,N);
	t[3] = tick() - t[3];
	
	
	t[4] = tick();
	free(a);
	free(b);
	t[4] = tick() - t[4];
	
	t[0] = tick() - t[0];
	
	informar(t, resultado, N, k1, k2);
	
	return 0;
}
