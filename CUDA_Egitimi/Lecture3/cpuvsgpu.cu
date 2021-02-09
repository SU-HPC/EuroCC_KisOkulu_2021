#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <ctime>

#define N (1024*1024)
#define M (10000)

__global__ void cudakernel(float *buf)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  float data = 1.0f * i / N;
  for(int j = 0; j < M; j++) {
    data = data * data - 0.25f;
  }
  buf[i] = data;
}

int main() {
  
  float data[N];
  memset(data, 0, sizeof(float)*N);

  //Here is the CPU part
  std::cout << "************************************\n";

  auto start = std::chrono::system_clock::now();
  for(int i = 0; i < N; i++) {
    float d = (1.0f * i) / N;
    for(int j = 0; j < M; j++) {
      d = d * d - 0.25f;
    }
    data[i] = d;
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << "s\n";
  std::cout << "data[1001] = " << data[1001] << std::endl;

  std::cout << "************************************\n";

  //Parallel CPU
  memset(data, 0, sizeof(float)*N);
  start = std::chrono::system_clock::now();
  int nt = omp_get_max_threads();
  printf("Number of threads %d\n", nt);

#pragma omp parallel for
  for(int i = 0; i < N; i++) {
    float d = (1.0f * i) / N;
    for(int j = 0; j < M; j++) {
      d = d * d - 0.25f;
    }
    data[i] = d;
  }

  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << "s\n";
  std::cout << "data[1001] = " << data[1001] << std::endl;

  std::cout << "************************************\n";

  //Here is the GPU part
  memset(data, 0, sizeof(float)*N);
  /*
    Kullanılacak cihazı belirle
  */


  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceID);
  printf("Device Number: %d\n", deviceID);
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

  //1: cihazda bellek ayir
 
  start = std::chrono::system_clock::now();

  //2: gpu kodunu çalıştır

  end = std::chrono::system_clock::now();

  //3. datayı ana makineye kopyala
  //4. cihazda ayrılan hafızayı serbest bırak

  elapsed_seconds = end-start;
  end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time) << "elapsed time: " << elapsed_seconds.count() << "s\n";
  std::cout << "data[1001] = " << data[1001] << std::endl;
}
