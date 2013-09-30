#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//#define CANTELEM 134217728 //128M
#define CANTELEM 2097152 //2M

//Para calcular tiempo
double dwalltime(){
		double sec;
		struct timeval tv;
		gettimeofday(&tv,NULL);
		sec = tv.tv_sec + tv.tv_usec/1000000.0;
		return sec;
}


int main (int argc, char ** argv){
	unsigned int * V;
	double timetick;
	double limit = sqrt(CANTELEM)+1; //no puede haber divisores más grandes que limit 
	timetick = dwalltime();
	int i,j;
	if (NULL == (V = malloc(CANTELEM * sizeof(unsigned int)))){
		printf("No hay memoria suficiente :(\n");
		return -1;
	}
	for (i = 0; i < CANTELEM - 1; i++){V[i] = i;}
	printf("Inicialización = %f\n", dwalltime() - timetick);
	
	timetick = dwalltime();
	for (i = 2; i < limit; i++){
		for (j = 0; j < CANTELEM; j++){
			if (V[j] != 0){
				if ((V[j] % i) == 0){
					V[j] = 0;
				}
			}
		}
	}
	printf("Criba completa = %f\n", dwalltime() - timetick);
	for (i = 0; i < CANTELEM; i++){
		if (V[i] != 0){
			//printf("%u es primo\n", V[i]);
		}
	}
	return 0;
}

