#include <stdio.h>
#include <stdlib.h>
typedef int data_t;
// Para medir los tiempos
double tick(){
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}

void * read_ppm5(char * filename, int * width, int * height, int * bpp){
	FILE * fp;
	void * buffer;
	char magick[3];
	fp = fopen(filename, "r");
	if (fp == NULL){
		return NULL;
	}
	fgets(magick, 3, fp);
	if (strcmp(magick, "P5")){
		printf("Not a P5 ppm file\n");
		return NULL;
	}
	fgetc(fp);
	fscanf(fp, "%d %d", width, height);
	fscanf(fp, "%d", bpp);
	if ((*bpp) < 256){(*bpp) = 1;} else{ (*bpp) = 2;}
	*bpp=1;
	// GUARDA: La comparacion tiene precedencia sobre la asignacion!!!
	// if (buffer = malloc((*width) * (*height) * (*bpp) + 2) == NULL) exit(73);
	buffer = malloc((*width) * (*height) * (*bpp));
	fread(buffer, ((*width) * (*height) * (*bpp)), 1, fp);
	fclose(fp);
	return buffer;
}
/*
void initmatrixa(int a[N][N], int n){
	unsigned int i, j;
	for (j=0; j<n; j++){
		for (i=0; i<n; i++){
			a[i][j]=i+j;
		}
	}
}
*/
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

/*void initmatrixb(int a[N][N], int n){
	unsigned int i, j;
	for (j=0; j<n; j++){
		for (i=0; i<n; i++){
			a[i][j]=i-j;
		}
	}
}*/

//  Función secuencial: MSE
double MSE(int *a, int *b, const unsigned int n) {
    unsigned int i, j;
    double total=0;
    for(i = 0; i < n*n; i++) {
    	total += ((a[i] - b[i])*((a[i] - b[i])));
    }
	total/=n*n;
	printf("total = %f\n", total);
	return total;
}

float verificar(int n, int k1, int k2){
/*k1=7
    k2=9
    n=8
    (n*(k1*k1+k2*k2))/(n**2.0)
  */
	return (n*(k1*k1+k2*k2))/(float)(n*n);
}

int main(int argc, char *argv[]){
	int * a;
	int * b;
	double t;
	const unsigned int N  = (argc >= 2) ? atoi(argv[1]) : 8;
	const unsigned int k1 = (argc >= 3) ? atoi(argv[2]) : 7;
	const unsigned int k2 = (argc >= 4) ? atoi(argv[3]) : 9;
	double resultado;
	t = tick();
	a = malloc(N*N*sizeof(int));
	b = malloc(N*N*sizeof(int));
	init_matrix(a, N, 0, k1);
	init_matrix(b, N, 1, k2);
	//a = read_ppm5("dos00064.pgm",&h, &w, &bbp);
	//b = read_ppm5("dos00065.pgm",&h, &w, &bbp);
	t = tick() - t;
	printf("Alocacion  y lectura de las matrices: %f\n", t);
	//b[0][0]=-22;
	t = tick();
	resultado = MSE(a,b,N);
	t = tick() - t;
	printf("Procesamiento MSE en matrices: %f\n", t);
	printf("\x1B[33mResultado final  =====>>>  %f\x1B[0m\n", resultado);
  	if (resultado == verificar (N, k1, k2))
    		printf("\x1B[32mVerificación: %f == %f\x1B[0m\n", resultado, verificar (N, k1, k2));
  	else
   		printf("\x1B[31mVerificación: %f == %f\x1B[0m\n", resultado, verificar (N, k1, k2));
	return 0;
}
