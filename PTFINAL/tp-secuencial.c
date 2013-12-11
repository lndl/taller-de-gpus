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

double cpu_exec(unsigned int n, unsigned int k1, unsigned int k2, double* t){
	data_t * a, * b;
	double resultado;
	
	t[0] = tick();
	
	t[1] = tick();
	a = malloc(n*n*sizeof(data_t));
	b = malloc(n*n*sizeof(data_t));
	t[1] = tick() - t[1];
	
	t[2] = tick();
	init_matrix(a, n, 0, k1);
	init_matrix(b, n, 1, k2);
	t[2] = tick() - t[2];
	
	
	t[3] = tick();
	resultado = MSE(a,b,n);
	t[3] = tick() - t[3];
	
	
	t[4] = tick();
	free(a);
	free(b);
	t[4] = tick() - t[4];
	
	t[0] = tick() - t[0];
	
	return resultado;
}

int main(int argc, char** argv)
{
  const unsigned int n      = (argc >= 2) ? atoi(argv[1]) : 8;
  const unsigned int count  = (argc >= 3) ? atoi(argv[2]) : 1;
  const unsigned int k1     = (argc >= 4) ? atoi(argv[3]) : 7;
  const unsigned int k2     = (argc >= 5) ? atoi(argv[4]) : 9;
  unsigned int i;
  double t_acc[10], resultado;

  array_init(t_acc, 10, 0);
  for(i=0; i<count; i++)
  {
    double t[10];
    array_init(t, 10, -1);
    resultado = cpu_exec(n,k1,k2,t);
    array_acc(t_acc, t, 10);
  }
  informar(t_acc, resultado, n, k1, k2);

  return 0;
}
