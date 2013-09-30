#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

//#define CANTELEM 134217728 //128M
#define CANTELEM 2097152 //2M

#define THREADS 4
pthread_t threads[THREADS];
pthread_mutex_t mutex;

unsigned int * V; //el arreglo con todos los numeros hasta CANTELEM

//Compartida por todos los procesos, cada uno toma un divisor
//e incrementa la variable para que el proximo tome el divisor siguiente.
int divisor = 2;
double limit;

//Para calcular tiempo
double dwalltime(){
		double sec;
		struct timeval tv;
		gettimeofday(&tv,NULL);
		sec = tv.tv_sec + tv.tv_usec/1000000.0;
		return sec;
}

void * criba(void * tid){
	int mydiv=0;
	int j;
	while (mydiv < limit){
		pthread_mutex_lock(&mutex);
		mydiv = divisor;
		divisor++;
		pthread_mutex_unlock(&mutex);
		for (j = 0; j < CANTELEM; j++){
			if (V[j] != 0){
				if ((V[j] % mydiv) == 0){
					V[j] = 0;
				}
			}
		}
	}
	printf("Thread %ld terminado, mydiv = %d\n", tid, mydiv);
	pthread_exit(NULL);
}


int main (int argc, char ** argv){
	double timetick;
	int i,j;
	limit = sqrt(CANTELEM) + 1; //No puede haber divisores mas grandes que limit	
	timetick = dwalltime();
	if (NULL == (V = malloc(CANTELEM * sizeof(unsigned int)))){
		printf("No hay memoria suficiente :(\n");
		return -1;
	}
	for (i = 0; i < CANTELEM - 1; i++){V[i] = i;}
	printf("Inicialización = %f\n", dwalltime() - timetick);
	
	//Threads init, los threads quedan parados hasta que el mutex se libera
	pthread_mutex_lock(&mutex);
	for (i = 0; i < THREADS; i++){
		pthread_create( &threads[i], NULL, criba, (void*)i);
	}
	pthread_mutex_unlock(&mutex);
	timetick = dwalltime();
	//Faltan los join
	for (i = 0; i < THREADS; i++){
		pthread_join( threads[i], NULL);
	}
	printf("Criba completa = %f\n", dwalltime() - timetick);

	for (i = 0; i < CANTELEM; i++){
		if (V[i] != 0){
	//		printf("%u es primo\n", V[i]);
		}
	}
	return 0;
}

