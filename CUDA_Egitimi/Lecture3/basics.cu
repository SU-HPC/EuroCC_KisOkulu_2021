#include <stdio.h>
#include <iostream>
#include "common.h"

__global__ void cudaKernel(int * data){
	int i = threadIdx.x + blockIdx.x * blockDim.x;	
	data[i] = threadIdx.x;
}

void print_arr(int * arr, int size){
	int i = 0;
	std::cout << "Arr = [";
	for(; i < size-1; i++){
		std::cout << arr[i] << ", ";
	}
	std::cout << arr[i] << "]" << std::endl;	
}
	

int main(){
	int device = 0, N = 24;
	int * h_data, * d_data;
	h_data = (int *)malloc(N * sizeof(int));
	cudaSetDevice(device);
	
	cudaMalloc(&d_data, N * sizeof(int));
	cudaMemcpy(d_data, h_data, N *sizeof(int), cudaMemcpyHostToDevice);
	dim3 gridDims(4);
	dim3 blockDims(6);
	cudaKernel<<<gridDims, blockDims>>>(d_data);
	cudaMemcpy(h_data, d_data, N *sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	print_arr(h_data, N);
	cudaFree(d_data);
	return 0;
}
