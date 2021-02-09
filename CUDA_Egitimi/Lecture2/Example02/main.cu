#include <cstdio>
#include <cstdlib>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

__global__ void kernel(int *a, int N) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  a[i]=i;
}

int main() {
  
  int N = 4097;
  int threads = 128;
  int blocks = (N+threads-1)/threads ;
  int *a;

  printf("No blocks: %d\n", blocks);
  
  gpuErrchk(cudaMallocManaged(&a, N*sizeof(int)));
    
  kernel<<<blocks, threads>>>(a, N);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());

  for(int i=0;i<10;i++) {
    printf("%d\n",a[i]);
  }

  cudaFree(a);
  return 0;
}
