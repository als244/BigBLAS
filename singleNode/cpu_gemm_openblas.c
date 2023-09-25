#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cblas.h>

#define M_PI 3.14159265358979323846

void sampleNormal(float * X, int size, float mean, float var) {
	float x, y, z, std, val;
	for (int i = 0; i < size; i++){
		x = (float)rand() / RAND_MAX;
		y = (float)rand() / RAND_MAX;
		z = sqrtf(-2 * logf(x)) * cosf(2 * M_PI * y);
		std = sqrtf(var);
		val = std * z + mean;
		X[i] = val;
	}
}

int main(int argc, char * argv[]){

	if (argc != 5){
		fprintf(stderr, "Wrong number of args\n");
		exit(1);
	}

	int m = atoi(argv[1]);
	int k = atoi(argv[2]);
	int n = atoi(argv[3]);

	int num_threads = atoi(argv[4]);

	if (m <= 0 || k <= 0 || n <= 0){
		fprintf(stderr, "Bad dimensions\n");
		exit(1);
	}

	float * A = malloc(m * k * sizeof(float));
	float * B = malloc(k * n * sizeof(float));
	float * C = malloc(m * n * sizeof(float));

	sampleNormal(A, m * k, 0.0, 1.0 / 20.0);
	sampleNormal(B, k * n, 0.0, 1.0 / 20.0);


	openblas_set_num_threads(num_threads);

	struct timeval  tv1, tv2;
    	gettimeofday(&tv1, NULL);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
					m, n, k, 1.0, A, k, B, n, 
					0.0, C, n);
  	gettimeofday(&tv2, NULL);;
    	double time_taken = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec); // in seconds

    	printf("SGEMM where: m=%d, k=%d, n=%d took --- %f seconds\n", m, k, n, time_taken);
    	printf("GFLOPS Proxy: %f\n", 2 * (float) m * (float) k * (float) n / time_taken / 1e9); 

    	free(A);
    	free(B);
    	free(C);

    	return 0;

}