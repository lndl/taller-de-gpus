#ifndef EXTRAS_H
#define EXTRAS_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef int data_t;

// Para medir los tiempos
double tick();
void array_init(double* a, int n, double k);
void array_acc(double* acc, double* b, int n);
// Funcion para la inicializacion de las matrices
void init_matrix(data_t *M, const unsigned int size, int orientation, int k );
// Verificacion de resultados
float verificar(int n, int k1, int k2);
// Informe final de tiempos y resultados
void informar(double* times, double resultado, int N, int k1, int k2);
// Impresion de matrices
void print_matrix(data_t * const M, const unsigned int size);

#endif
