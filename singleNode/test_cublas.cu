#include <stdio.h>
#include <stdlib.h>
#include <cublasLt.h>
#include <curand.h>
#include <sys/time.h>

struct matrixMeta {
	size_t rows;
	size_t cols;
	int typeID;
};

int main(int argc, char * argv[]){

	if (argc != 4){
		fprintf(stderr, "Wrong number of args\n");
		exit(1);
	}

	char * fileNameA = argv[1];
	char * fileNameB = argv[2];
	char * fileNameC = argv[3];

	char * pathA, *pathAMeta, *pathB, *pathBMeta;

	asprintf(&pathA, "/home/shein/Documents/grad_school/research/BigBLAS/data/%s", fileNameA);
	asprintf(&pathAMeta, "/home/shein/Documents/grad_school/research/BigBLAS/data/%s.metadata", fileNameA);
	asprintf(&pathB, "/home/shein/Documents/grad_school/research/BigBLAS/data/%s", fileNameB);
	asprintf(&pathBMeta, "/home/shein/Documents/grad_school/research/BigBLAS/data/%s.metadata", fileNameB);

	FILE * fpA, *fpAMeta, * fpB, * fpBMeta;

	struct matrixMeta A_metadata, B_metadata;

	// get A metadata
	fpAMeta = fopen(pathAMeta, "r");
	free(pathAMeta);
	fread(&A_metadata, sizeof(struct matrixMeta), 1, fpAMeta);
	fclose(fpAMeta);

	// get B metadata
	fpBMeta = fopen(pathBMeta, "r");
	free(pathBMeta);
	fread(&B_metadata, sizeof(struct matrixMeta), 1, fpBMeta);
	fclose(fpBMeta);

	size_t M = A_metadata.rows;
	size_t K = A_metadata.cols;
	size_t N = B_metadata.cols;

	if (K != B_metadata.rows){
		fprintf(stderr, "Matrix dims do not align!\n");
		exit(1);
	}

	// Create files for output C matrix
	char * pathC, * pathCMeta;
	asprintf(&pathC, "/home/shein/Documents/grad_school/research/BigBLAS/data/output_%s", fileNameC);
	asprintf(&pathCMeta, "/home/shein/Documents/grad_school/research/BigBLAS/data/output_%s.metadata", fileNameC);

	FILE * fpC, *fpCMeta;

	// write metadata struct
	struct matrixMeta C_metadata = {M, N, A_metadata.typeID};
	fpCMeta = fopen(pathCMeta, "w+");
	free(pathCMeta);
	fwrite(&C_metadata, sizeof(struct matrixMeta), 1, fpCMeta);
	fclose(fpCMeta);


	// MATRICES SMALL ENOUGHT TO READ INTO MEMORY!
	float * A, *B;

	A = (float *) calloc(M * K, sizeof(float));
	B = (float *) calloc(K * N, sizeof(float));
	
	// read A
	fpA = fopen(pathA, "r");
	free(pathA);
	fread(A, sizeof(float), M * K, fpA);
	fclose(fpA);

	// read B
	fpB = fopen(pathB, "r");
	free(pathB);
	fread(B, sizeof(float), K * N, fpB);
	fclose(fpB);



	/* SETTING CUBLAS UP */
	void *d_A, *d_B, *d_C;

	cudaMalloc(&d_A, M * K * sizeof(float));
	cudaMalloc(&d_B, K * M * sizeof(float));
	cudaMalloc(&d_C, M * N * sizeof(float));


	// copy over input matrices to device
	cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

	// free the input matrices
	free(A);
	free(B);

	// allocate output matrix
	float * C = (float *) calloc(M * N, sizeof(float));

	
	// deal with cuBLAS structs
	cublasStatus_t status;
	cublasLtHandle_t handle;
	status = cublasLtCreate(&handle);


	cublasOperation_t transa = CUBLAS_OP_T;
	cublasOperation_t transb = CUBLAS_OP_N;

	cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;

	cublasLtMatmulDesc_t matmulDesc;

	status = cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F);
	status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
	status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

	// A Transposed (from row-major to column-major), not B/D (but still held in col-major format internally)
	// m and k must be multiples of 4, perferablly multiples of 16
	status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, K, M, K);
	status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, K);
	status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M);
	status = cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, M, N, M);


	cublasLtMatmulPreference_t pref;
	status = cublasLtMatmulPreferenceCreate(&pref);
	// ALLOW workspace mem...
	const size_t workspaceBytes = 0;
	status = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes));

	int algoCount = 1;
	int retAlgoCount = 0;

	cublasLtMatmulHeuristicResult_t heuristicResultsArray = {};

	status = cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, algoCount, &heuristicResultsArray, &retAlgoCount);

	cublasLtMatmulAlgo_t algo = heuristicResultsArray.algo;

	//void * workspace;
	void * workspace = NULL;
	cudaMalloc(&workspace, workspaceBytes);


	// PREFORM MATMUL ON GPU

	float alpha = 1.0, beta = 0.0;
	struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);

    cudaDeviceSynchronize();

	status = cublasLtMatmul(handle,
							matmulDesc,
							&alpha,
							d_A,
							Adesc,
							d_B,
							Bdesc,
							&beta,
							d_C,
							Cdesc,
							d_C,
							Ddesc,
							&algo,
							workspace,
							workspaceBytes,
							0);


	cudaDeviceSynchronize();

	gettimeofday(&tv2, NULL);
    double time_taken = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         					(double) (tv2.tv_sec - tv1.tv_sec); // in seconds

    printf("SGEMM where: m=%zu, k=%zu, n=%zu took --- %f seconds\n", M, K, N, time_taken);


    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    fpC = fopen(pathC, "w+");
	free(pathC);
	fwrite(C, sizeof(float), M * N, fpC);
	fclose(fpC);


    /* CLEAN UP MEMORY */

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
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(C);

	return 0;

}