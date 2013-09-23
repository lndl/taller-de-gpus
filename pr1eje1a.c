#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CANTELEM 134217728
#define CCONST 32
//Para calcular tiempo
double dwalltime(){
		double sec;
		struct timeval tv;
		gettimeofday(&tv,NULL);
		sec = tv.tv_sec + tv.tv_usec/1000000.0;
		return sec;
}


int main (int argc, char ** argv){
	int C;
	unsigned int * V;
	double timetick;
	//inicializacion; ojo que el entero hace overflow en la mitad del arreglo
	timetick = dwalltime();
	int i;
	if (NULL == (V = malloc(CANTELEM * sizeof(unsigned int)))){
		printf("No hay memoria suficiente :(\n");
		return -1;
	}
	for (i = 0; i < CANTELEM - 1; i++){V[i] = i;}
	printf("Inicialización = %f\n", dwalltime() - timetick);
	
	timetick = dwalltime();
	for (i = 0; i < CANTELEM-1; i++){
		V[i] = V[i]*CCONST;
	}
	printf("Multiplicación = %f\n", dwalltime() - timetick);

	//Te debo el chequeo :P
	return 0;
}
