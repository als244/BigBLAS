#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <cblas.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>


#define NUM_THREADS 16

struct matrixMeta {
	size_t rows;
	size_t cols;
	int typeID;
};

bool SAVE_TEMP_PARTIAL = false;
bool GPU_ENABLED = true;
int STRAT_ID = 0;

long roundUpToMultiple(long numToRound, long multiple)
{
    long remainder = numToRound % multiple;
    
    if (remainder == 0){
        return numToRound;
    }

    return numToRound + multiple - remainder;
}

// assume block inds start at 0 on top left, then left to right and top to bottom

// TODO: can load blocks more efficiently if each subblock is saved contiguously (iterating over each row means slower disk access)
void load_subblock(float * subBlock, size_t blockInd, size_t subRows, size_t subCols, FILE * fpBigMatrix, size_t rows, size_t cols, float * devSubBlock){
	size_t subBlocksInRow = cols / subCols;

	size_t subBlockRowStart = (blockInd / subBlocksInRow) * subRows;
	size_t subBlockColStart = (blockInd % subBlocksInRow) * subCols;

	size_t totalOff;
	for (int rowId = 0; rowId < subRows; rowId++){
		totalOff = ((subBlockRowStart + rowId) * cols + subBlockColStart) * sizeof(float);
		fseek(fpBigMatrix, totalOff, SEEK_SET);
		fread(subBlock + rowId * subCols, sizeof(float), subCols, fpBigMatrix);
	}

	// if we are loading into device
	if (devSubBlock){
		cudaMemcpy(devSubBlock, subBlock, subRows * subCols * sizeof(float), cudaMemcpyHostToDevice);
	}
}

// TODO: would be faster to save subblocks contiguously and accumuate/piece together after...
void save_subblock(float * subBlock, size_t blockInd, size_t subRows, size_t subCols, FILE * fpBigMatrix, size_t rows, size_t cols, float * devSubBlock){
	size_t subBlocksInRow = cols / subCols;
	size_t subBlockRowStart = (blockInd / subBlocksInRow) * subRows;
	size_t subBlockColStart = (blockInd % subBlocksInRow) * subCols;

	size_t totalOff;

	// if we are saving from device
	if (devSubBlock){
		cudaMemcpy(subBlock, devSubBlock, subRows * subCols * sizeof(float), cudaMemcpyDeviceToHost);
	}

	for (int rowId = 0; rowId < subRows; rowId++){
		totalOff = ((subBlockRowStart + rowId) * cols + subBlockColStart) * sizeof(float);
		fseek(fpBigMatrix, totalOff, SEEK_SET);
		fwrite(subBlock + rowId * subCols, sizeof(float), subCols, fpBigMatrix);
	}
}

void save_partial_result(float * subBlock, size_t blockInd, size_t partialInd, size_t subRows, size_t subCols){
	
	FILE * fpPartial;
	char * partialPath;
	asprintf(&partialPath, "/mnt/storage/data/BigBLAS/temp/%zu_%zu", blockInd, partialInd);
	
	fpPartial = fopen(partialPath, "w+");
	free(partialPath);
	fwrite(subBlock, sizeof(float), subRows * subCols, fpPartial);
	fclose(fpPartial);
}

int main(int argc, char * argv[]){

	char * fileNameA = argv[1];
	char * fileNameB = argv[2];

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

	size_t m = A_metadata.rows;
	size_t k = A_metadata.cols;
	size_t n = B_metadata.cols;

	if (k != B_metadata.rows){
		fprintf(stderr, "Matrix dims do not align!\n");
		exit(1);
	}

	// for now do nothing with these...
	// ASSUME A_type = B_type
	int A_type = A_metadata.typeID;
	int B_type = B_metadata.typeID;


	// Create files for output C matrix
	char * pathC, * pathCMeta;
	asprintf(&pathC, "/home/shein/Documents/grad_school/research/BigBLAS/data/output_%s_%s", fileNameA, fileNameB);
	asprintf(&pathCMeta, "/home/shein/Documents/grad_school/research/BigBLAS/data/output_%s_%s.metadata", fileNameA, fileNameB);

	FILE * fpC, *fpCMeta;

	// write metadata struct
	struct matrixMeta C_metadata = {m, n, A_type};
	fpCMeta = fopen(pathCMeta, "w+");
	free(pathCMeta);
	fwrite(&C_metadata, sizeof(struct matrixMeta), 1, fpCMeta);
	fclose(fpCMeta);

	// TODO: get formula to compute instead of manual
	size_t subM = 32768;
	size_t subK = 65536;
	size_t subN = 32768;

	float *subA, *subB, *subC;

	subA = (float *) calloc(subM * subK, sizeof(float));
	subB = (float *) calloc(subK * subN, sizeof(float));
	subC = (float *) calloc(subM * subN, sizeof(float));


	// TODO: will worry about partial blocks / zero-padding later
	size_t blocksM = round((double) m / (double) subM);
	size_t blocksK = round((double) k / (double) subK);
	size_t blocksN = round((double) n / (double) subN);


	size_t blocksA = blocksM * blocksK;
	size_t blocksB = blocksK * blocksN;
	size_t blocksC = blocksM * blocksN;

	// OPEN BIG MATRIX FILES FOR READ/WRITING
	fpA = fopen(pathA, "w+");
	free(pathA);
	fpB = fopen(pathB, "w+");
	free(pathB);
	fpC = fopen(pathC, "w+");
	free(pathC);

	/* SETTING CUBLAS STUFF */
	void *d_subA, *d_subB, *d_subC;

	if (GPU_ENABLED){
		cudaMalloc(&d_subA, subM * subK * sizeof(float));
		cudaMalloc(&d_subB, subK * subM * sizeof(float));
		cudaMalloc(&d_subC, subM * subN * sizeof(float));
	}

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
	status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, subK, subM, subK);
	status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, subK, subN, subK);
	status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, subM, subN, subM);
	status = cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, subM, subN, subM);


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

	/* STARTING MATRIX MULTIPLICATIONS! */
	openblas_set_num_threads(NUM_THREADS);

	size_t cnt, aInd, bInd, cInd;
	float alpha = 1.0;
	float beta;

	size_t matMulCnt = 0;


	struct timeval start, end, curTime;
	double time_taken;
    gettimeofday(&start, NULL);
	
	/* STRATEGY 1: Keeping same sublock of A in memory and exhausting matmuls paired with it */
	// for simplicity iterate keeping each of the "A blocks" in memory and pairing them will all the "B blocks"
	// creates a lot of temp results that take up storage and need to be aggregated

	if (STRAT_ID == 0){
		size_t partialInd;
		for (aInd = 0; aInd < blocksA; aInd++){
			load_subblock(subA, aInd, subM, subK, fpA, m, k, d_subA);
			cnt = 0;
			bInd = aInd % blocksK;
			while (cnt < blocksN){
				// load b
				load_subblock(subB, bInd, subK, subN, fpB, k, n, d_subB);
				
				
				cInd = (aInd / blocksK) * blocksN + cnt;

				if (cnt == 0 || SAVE_TEMP_PARTIAL){
					beta = 0;
				}
				else{
					load_subblock(subC, cInd, subM, subN, fpC, m, n, d_subC);
					beta = 1;
				}

				// do matmul
				if (GPU_ENABLED){
					status = cublasLtMatmul(handle,
							matmulDesc,
							&alpha,
							d_subA,
							Adesc,
							d_subB,
							Bdesc,
							&beta,
							d_subC,
							Cdesc,
							d_subC,
							Ddesc,
							&algo,
							workspace,
							workspaceBytes,
							0);
				}
				else{
					cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
						subM, subN, subK, 1.0, subA, subK, subB, subN, 
						beta, subC, subN);
				}


				matMulCnt += 1;
				if (matMulCnt % 1 == 0){
					gettimeofday(&curTime, NULL);
					time_taken = (double) (curTime.tv_usec - start.tv_usec) / 1000000 +
        							(double) (curTime.tv_sec - start.tv_sec);
					printf("After %zu matMuls: %f seconds\n", matMulCnt, time_taken);
				}
				
				// save partial result of C
				
				/* saving partial result to disk to aggregate later */
				if (SAVE_TEMP_PARTIAL){
					partialInd = aInd % blocksK;
					save_partial_result(subC, cInd, partialInd, subM, subN);
				}
				// saving to block in local filesystem
				else{
					save_subblock(subC, cInd, subM, subN, fpC, m, n, d_subC);
				}

				// get next B ind
				bInd += blocksK;

				// increment cnt
				cnt += 1;
			}
		}
	}


	/* STRATEGY 2: Keeping same partial result block in memory and iterating over result blocks */
	// iterate over each block in the output by keeping each "C block" in memory
	
	matMulCnt = 0;

	if (STRAT_ID == 1){

		for (cInd = 0; cInd < blocksC; cInd++){
			// get starting inds for A row and B col
			aInd = (cInd / blocksN) * blocksK;
			bInd = (cInd % blocksN) * blocksK;

			cnt = 0;
			beta = 0;
			while (cnt < blocksK){
				// load subA
				load_subblock(subA, aInd, subM, subK, fpA, m, k, d_subA);
				// load subB
				load_subblock(subB, bInd, subK, subN, fpB, k, n, d_subB);

				// do matmul 
				// (beta = 0 for first iteration of inner loop then 1 because adding to prior results)

				if (GPU_ENABLED){
					status = cublasLtMatmul(handle,
							matmulDesc,
							&alpha,
							d_subA,
							Adesc,
							d_subB,
							Bdesc,
							&beta,
							d_subC,
							Cdesc,
							d_subC,
							Ddesc,
							&algo,
							workspace,
							workspaceBytes,
							0);
				}
				else{
					cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
						subM, subN, subK, 1.0, subA, subK, subB, subN, 
						beta, subC, subN);
				}
				matMulCnt += 1;
				if (matMulCnt % 1 == 0){
					gettimeofday(&curTime, NULL);
					time_taken = (double) (curTime.tv_usec - start.tv_usec) / 1000000 +
        							(double) (curTime.tv_sec - start.tv_sec);
					printf("After %zu matMuls: %f seconds\n", matMulCnt, time_taken);
				}

				// get next block in row of A
				aInd += 1;
				// get next block in col of B
				bInd += blocksK;

				// do next pairing of sub row/col blocks
				beta = 1;
				cnt += 1;
			}

			// save output subblock of C
			save_subblock(subC, cInd, subM, subN, fpC, m, n, d_subC);

		}
	}

	gettimeofday(&end, NULL);
    time_taken = (double) (end.tv_usec - start.tv_usec) / 1000000 +
        (double) (end.tv_sec - start.tv_sec); // in seconds

    printf("\n\n\nSGEMM where: m=%zu, k=%zu, n=%zu took --- %f seconds\n", m, k, n, time_taken);

	free(subA);
	free(subB);
	free(subC);

	cudaFree(d_subA);
	cudaFree(d_subB);
	cudaFree(d_subC);

	fclose(fpA);
	fclose(fpB);
	fclose(fpC);






}