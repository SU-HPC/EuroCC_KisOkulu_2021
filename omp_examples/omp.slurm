#!/bin/bash
#SBATCH -A takyildiz #your_user_account
#SBATCH -J omp-example
#SBATCH -N 1 #number of nodes to be used
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -p debug #name of the job queue
#SBATCH --time=00:15:00

echo "***********************"
echo "0) False sharing: "
g++ -o pi 0_pi_omp_false_sharing.c -fopenmp
srun ./pi 
#echo "***********************"
#echo "1) Padding: "
#g++ -o pi_padded 1_pi_omp_padded.c -fopenmp
#srun ./pi_padded
#echo "***********************"
#echo "2) Locking: "
#g++ -o pi_lock 2_pi_omp_lock.c -fopenmp
#srun ./pi_lock
#echo "***********************"
#echo "3) Critical: "
#g++ -o pi_critical 3_pi_omp_critical.c -fopenmp
#srun ./pi_critical
#echo "***********************"
#echo "4) Critical Opt: "
#g++ -o pi_critical 4_pi_omp_critical.c -fopenmp
#srun ./pi_critical
#echo "***********************"
#echo "5) Atomic: "
#g++ -o pi_atomic 5_pi_omp_atomic.c -fopenmp
#srun ./pi_atomic
#echo "***********************"
#echo "6) Reduction: "
#g++ -o pi_reduction 6_pi_omp_loop.c -fopenmp
#srun ./pi_reduction
#echo "***********************"
exit
