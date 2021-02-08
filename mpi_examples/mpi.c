#include <mpi.h>
#include <stdio.h>

//#define REDUCE

static long long num_steps=1000000000;
double step;
int main(int argc, char** argv){
	int i, myid, num_procs;
	double x, pi=0, remote_sum, sum=0, start=0, end=0, all_sum=0;

	/*
		1. MPI ortamını başlat
		2. process id'yi doldur
		3. total process sayısını doldur
	*/

	start = MPI_Wtime();
	step = 1.0/(double) num_steps;
	for (i = myid; i< num_steps; i=i+num_procs){
		x =(i+0.5)*step;
		sum +=4.0/(1.0+x*x);
	}
#ifdef REDUCE
	printf("All_reduce:\n");
	MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	pi *= step;
#else
	printf("Send-recieve:\n");
	if (myid==0){
		for (i = 1; i< num_procs;i++){
			MPI_Status status;
			/*
				mesaj bekle
			*/
			sum +=remote_sum;
		}
		pi=sum*step;
	} else {
		/*
			mesaj yolla
		*/
	}
#endif
	/*
		mpi ortamını bitir
	*/

	if (myid ==0){
		end = MPI_Wtime();
		printf("Processors %d, took %f, value: %lf\n", num_procs, end-start, pi);
	}
	return 0;
}
