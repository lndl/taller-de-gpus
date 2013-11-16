#include <stdio.h>
#define N 8

void initmatrixa(int a[N][N], int n){
	unsigned int i, j;
	for (j=0; j<n; j++){
		for (i=0; i<n; i++){
			a[i][j]=i+j;
		}
	}
}

void initmatrixb(int a[N][N], int n){
	unsigned int i, j;
	for (j=0; j<n; j++){
		for (i=0; i<n; i++){
			a[i][j]=i-j;
		}
	}
}

//  Función secuencial: MSE
void MSE(int a[N][N], int b[N][N], const unsigned int n) {
    unsigned int i, j;
    double total=0;
    for(i = 0; i < n; i++) {
        for (j = 0; j < n; j++){
            total += ((a[i][j] - b[i][j])/n)*((a[i][j] - b[i][j])/n);
        }
    }
	printf("total = %f\n", total);
}

int main(){
	int a[N][N];
	int b[N][N];
	initmatrixa(a, N);
	initmatrixa(b, N);
	b[0][0]=-22;
	MSE(a,b,N);
	return 0;
}
