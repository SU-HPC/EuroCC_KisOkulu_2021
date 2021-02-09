#!/bin/bash
##SBATCH -A takyildiz #your_user_account
#SBATCH -J cuda-example
#SBATCH -N 1 #number of nodes to be used
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -p debug #name of the job queue
#SBATCH --time=00:15:00

nvcc -o basics basics.cu --std=c++11
srun ./basics
#nvcc -o dimension dimensions.c
#srun ./dimension
#nvcc -o cpuvsgpu cpuvsgpu.cu -Xcompiler -fopenmp -Xcompiler -O3 -O3
#srun ./cpuvsgpu
