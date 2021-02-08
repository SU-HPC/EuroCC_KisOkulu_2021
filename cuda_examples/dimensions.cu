#include <stdio.h>
#include <omp.h>
#include "common.h"

__global__ void dummy() {
  unsigned long long int target = 1000000;
  
  unsigned long long int noBlocks = /*_ _ _ _ _ _ _*/;
  unsigned long long int noThreadsInBlock = /*_ _ _ _ _ _ _*/;
  unsigned long long int noTotalThreads = /*_ _ _ _ _ _ _*/;
  
  // unique block index inside a 3D block grid
  unsigned long long int blockID = /*_ _ _ _ _ _ _*/;
  
  unsigned long long int localThreadID = /*_ _ _ _ _ _ _*/;
  
  unsigned long long int globalThreadID = /*_ _ _ _ _ _ _*/;
  
  if(localThreadID == target) {
    printf("Total no blocks:       %ld\n", noBlocks);
    printf("No threads in a block: %ld\n", noThreadsInBlock);
    printf("Total no threads:      %ld\n", noTotalThreads);
    printf("---------------------------------------------------\n");
  }
  
  if(globalThreadID == target) {
    printf("Hello from block: %ld\n", blockID);
    printf("\tGrid dimensions are (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
    printf("\tBlock coords.   are (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("---------------------------------------------------\n");

    printf("Local thread ID:  %ld\n", localThreadID);
    printf("Global thread ID: %ld\n", globalThreadID);
    printf("\tBlock dimensions are (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("\tThread coords.   are (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
  }
}

int main() {
  int device = 0;
  cudaSetDevice(device);
  
  /******************************************************************************/
  /* Getting the device properties here */
  //  cudaGetDevice(&device);
  
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
  printf("  Available memory in MBs: %f\n", double(prop.totalGlobalMem) / 1e+6);
  /******************************************************************************/

  dim3 gridDims(256, 256, 256);
  dim3 blockDims(2, 16, 32);
  dummy<<<gridDims, blockDims>>>();
  cudaDeviceSynchronize();
  cudaCheck(cudaPeekAtLastError());

  return 0;
} /* end main */
