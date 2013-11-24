/*
** Hello World using CUDA
** 
** The string "Hello World!" is mangled then restored using a common CUDA idiom
**
** Byron Galbraith
** 2009-02-18
*/
#include <cuda.h>
#include <stdio.h>

// Prototypes
__global__ void helloWorld(char*);
void devicenfo(void);

// Host function
int
main(int argc, char** argv)
{
  int i;

  //Prints out device info
  devicenfo();

  // desired output
  char str[] = "Hello World!";

  // mangle contents of output
  // the null character is left intact for simplicity
  for(i = 0; i < 12; i++)
    str[i] -= i;

  // allocate memory on the device 
  char *d_str;
  size_t size = sizeof(str);
  cudaMalloc((void**)&d_str, size);

  // copy the string to the device
  cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);

  // set the grid and block sizes
  dim3 dimGrid(2);   // one block per word  
  dim3 dimBlock(6); // one thread per character
  
  // invoke the kernel
  helloWorld<<< dimGrid, dimBlock >>>(d_str);

  // retrieve the results from the device
  cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost);

  // free up the allocated memory on the device
  cudaFree(d_str);
  
  // everyone's favorite part
  printf("%s\n", str);

  return 0;
}

// Device kernel
__global__ void
helloWorld(char* str)
{
  // determine where in the thread grid we are
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // unmangle output
  str[idx] += idx;
}

// Device info
void 
devicenfo(void)
{
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
