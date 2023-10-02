#include <stdio.h>
#include <stdlib.h>
#include <cublasLt.h>
#include <curand.h>
#include <sys/time.h>


int main(int argc, char * argv[]){

	if (argc != 4){
		fprintf(stderr, "Wrong number of args\n");
		exit(1);
	}

	size_t m = atol(argv[1]);
	size_t k = atol(argv[2]);
	size_t n = atol(argv[3]);

	if (m <= 0 || k <= 0 || n <= 0){
		fprintf(stderr, "Bad dimensions\n");
		exit(1);
	}

	// using cuRand to populate matrices
	curandGenerator_t gen;
	curandStatus_t curand_status;
	curand_status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curand_status = curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	// Init Matrices
	float * A, *B, *D;
	cudaMalloc(&A, m * k * sizeof(float));
	cudaMalloc(&B, k * n * sizeof(float));
	cudaMalloc(&D, m * n * sizeof(float));

	cudaDeviceSynchronize();

	if ((A == NULL) || (B == NULL) || (D == NULL)){
		fprintf(stderr, "OUT OF MEMORY!\n");
		exit(1);
	}

	// choosing arbitrary stddev that might resemeble init weights of Neural Network
	curand_status = curandGenerateNormal(gen, A, (size_t) (m * k), 0, sqrtf(1.0 / 20.0));
	curand_status = curandGenerateNormal(gen, B, (size_t) (k * n), 0, sqrtf(1.0 / 20.0));

	curand_status = curandDestroyGenerator(gen);

	// PERFORM MATMUL
	cublasStatus_t status;
	cublasLtHandle_t handle;
	status = cublasLtCreate(&handle);


	cublasOperation_t transa = CUBLAS_OP_T;
	cublasOperation_t transb = CUBLAS_OP_N;

	cublasLtMatrixLayout_t Adesc;
	cublasLtMatrixLayout_t Bdesc;
	cublasLtMatrixLayout_t Cdesc;
	cublasLtMatrixLayout_t Ddesc;


	cublasLtMatmulDesc_t matmulDesc;

	status = cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F);
	status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
	status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

	// A Transposed (from row-major to column-major), not B/D (but still held in col-major format internally)
	// m and k must be multiples of 4, perferablly multiples of 16
	status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, k, m, k);
	status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k, n, k);
	status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, m);
	status = cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, m, n, m);


	cublasLtMatmulPreference_t pref;
	status = cublasLtMatmulPreferenceCreate(&pref);
	// ALLOW 1 GB of workspace mem...
	//const size_t workspaceBytes = 1000000000;
	const size_t workspaceBytes = 0;
	//status = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes));

	int algoCount = 1;
	int retAlgoCount = 0;

	cublasLtMatmulHeuristicResult_t heuristicResultsArray = {};


	status = cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, algoCount, &heuristicResultsArray, &retAlgoCount);

	cublasLtMatmulAlgo_t algo = heuristicResultsArray.algo;

	//void * workspace;
	void * workspace = NULL;
	cudaMalloc(&workspace, workspaceBytes);

	float alpha = 1, beta = 0;


	struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);

    cudaDeviceSynchronize();
	
	status = cublasLtMatmul(handle,
							matmulDesc,
							&alpha,
							A,
							Adesc,
							B,
							Bdesc,
							&beta,
							NULL,
							Cdesc,
							D,
							Ddesc,
							&algo,
							workspace,
							workspaceBytes,
							0);

	cudaDeviceSynchronize();

	gettimeofday(&tv2, NULL);
    double time_taken = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         					(double) (tv2.tv_sec - tv1.tv_sec); // in seconds

    printf("SGEMM where: m=%zu, k=%zu, n=%zu took --- %f seconds\n", m, k, n, time_taken);
    printf("GFLOPS Proxy: %f\n", 2 * (double) m * (double) k * (double) n / time_taken / 1e9);
	
	
	// FREE workspace
	cudaFree(workspace);


	// FREE cuBlasLt Structs 

	status = cublasLtMatmulPreferenceDestroy(pref);
	status = cublasLtMatmulDescDestroy(matmulDesc);

	status = cublasLtMatrixLayoutDestroy(Adesc);
	status = cublasLtMatrixLayoutDestroy(Bdesc);
	status = cublasLtMatrixLayoutDestroy(Cdesc);
	status = cublasLtMatrixLayoutDestroy(Ddesc);

	status = cublasLtDestroy(handle);


	// FREE MATRICES

	cudaFree(A);
	cudaFree(B);
	cudaFree(D);

}