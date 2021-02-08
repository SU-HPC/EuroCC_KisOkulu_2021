#include <mpi.h>
#include <stdio.h>

//#define ALL_REDUCE

static int num_steps=1000;
int main(int argc, char** argv){
	int i, j, myid, num_procs, ms, id_sum=0;
	double start=0, end=0;
	MPI_Init(&argc, &argv);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	start = MPI_Wtime();
	for(i = 0; i < num_steps; i++){
#ifdef REDUCE
		MPI_Reduce(&myid, &id_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
		if (myid==0){
			id_sum=0;
			for (j = 1; j< num_procs;j++){
				MPI_Status status;
				MPI_Recv(&ms, 1, MPI_INT, j, 0, MPI_COMM_WORLD, &status);
				id_sum +=ms;
			}
		} else {
			MPI_Send(&myid, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
#endif
	}
	MPI_Finalize();

	if (myid ==0){
		end = MPI_Wtime();
#ifdef ALL_REDUCE
		printf("Reduce: ");
#else
		printf("Send & Receive: ");
#endif
		
		printf("Processors %d, took %f, value: %d\n", num_procs, end-start, id_sum);
	}
	return 0;
}
