
#include <stdio.h>
#include <stdlib.h>


// Declaración de función para ver recursos del device
void devicenfo(void);

int main(int argc, char *argv[]){
	devicenfo();
	return(0);
}


//  Sacar por pantalla información del *device*
void devicenfo(void){
  struct cudaDeviceProp capabilities;

  cudaGetDeviceProperties (&capabilities, 0);

  printf("->CUDA Platform & Capabilities\n");
  printf("Name: %s\n", capabilities.name);
  printf("totalGlobalMem: %.2f MB\n", capabilities.totalGlobalMem/1024.0f/1024.0f);
  printf("sharedMemPerBlock: %.2f KB\n", capabilities.sharedMemPerBlock/1024.0f);
  printf("regsPerBlock (32 bits): %d\n", capabilities.regsPerBlock);
  printf("warpSize: %d\n", capabilities.warpSize);
  printf("memPitch: %.2f KB\n", capabilities.memPitch/1024.0f);
  printf("maxThreadsPerBlock: %d\n", capabilities.maxThreadsPerBlock);
  printf("maxThreadsDim: %d x %d x %d\n", capabilities.maxThreadsDim[0], 
	 capabilities.maxThreadsDim[1], capabilities.maxThreadsDim[2]);
  printf("maxGridSize: %d x %d\n", capabilities.maxGridSize[0], 
	 capabilities.maxGridSize[1]);
  printf("totalConstMem: %.2f KB\n", capabilities.totalConstMem/1024.0f);
  printf("major.minor: %d.%d\n", capabilities.major, capabilities.minor);
  printf("clockRate: %.2f MHz\n", capabilities.clockRate/1024.0f);
  printf("textureAlignment: %d\n", capabilities.textureAlignment);
  printf("deviceOverlap: %d\n", capabilities.deviceOverlap);
  printf("multiProcessorCount: %d\n", capabilities.multiProcessorCount);
}
