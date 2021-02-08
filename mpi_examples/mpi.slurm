#!/bin/bash
#SBATCH -A takyildiz #your_user_account
#SBATCH -J mpi-example
#SBATCH -N 4 #number of nodes to be used
#SBACTH --ntasks-per-node=8
#SBATCH -c 1 #number of cores per task.. Not mandatory
#SBATCH -p debug
#SBATCH --time=00:15:00

module unload centos7.3/lib/openmpi/1.8.8-gcc-4.8.5
module load centos7.3/lib/openmpi/1.8.8-gcc-4.8.5
#mpicc -o pi_mpi mpi.c
#mpirun ./pi_mpi 
#mpicc -o pi_mpi mpi.c -DREDUCE
#mpirun ./pi_mpi 
mpicc -o rVSsr reduceVSsendrecv.c
mpirun ./rVSsr 
mpicc -o rVSsr reduceVSsendrecv.c -DREDUCE
mpirun ./rVSsr 
exit
