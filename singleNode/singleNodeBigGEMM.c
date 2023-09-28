#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cblas.h>

struct matrixMeta {
	size_t rows;
	size_t cols;
	int typeID;
};

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
void load_subblock(float * subBlock, size_t blockInd, size_t subRows, size_t subCols, FILE * fpBigMatrix, size_t rows, size_t cols){
	size_t subBlocksInRow = cols / subCols;

	size_t subBlockRowStart = (blockInd / subBlocksInRow) * subRows;
	size_t subBlockColStart = (blockInd % subBlocksInRow) * subCols;

	size_t totalOff;
	for (int rowId = 0; rowId < subRows; rowId++){
		totalOff = ((subBlockRowStart + rowId) * cols + subBlockColStart) * sizeof(float);
		fseek(fpBigMatrix, totalOff, SEEK_SET);
		fread(subBlock + rowId * subCols, sizeof(float), subCols, fpBigMatrix);
	}
}

// TODO: would be faster to save subblocks contiguously and accumuate/piece together after...
void save_subblock(float * subBlock, size_t blockInd, size_t subRows, size_t subCols, FILE * fpBigMatrix, size_t rows, size_t cols){
	size_t subBlocksInRow = cols / subCols;
	size_t subBlockRowStart = (blockInd / subBlocksInRow) * subRows;
	size_t subBlockColStart = (blockInd % subBlocksInRow) * subCols;

	size_t totalOff;
	for (int rowId = 0; rowId < subRows; rowId++){
		totalOff = ((subBlockRowStart + rowId) * cols + subBlockColStart) * sizeof(float);
		fseek(fpBigMatrix, totalOff, SEEK_SET);
		fwrite(subBlock + rowId * subCols, sizeof(float), subCols, fpBigMatrix);
	}
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
	size_t subM = 4096;
	size_t subK = 4096;
	size_t subN = 4096;

	float *subA, *subB, *subC;

	subA = (float *) calloc(subM * subK * sizeof(float));
	subB = (float *) calloc(subK * subN * sizeof(float));
	subC = (float *) calloc(subM * subN * sizeof(float));


	// TODO: will worry about partial blocks / zero-padding later
	size_t blocksM = round((double) M / (double) subM);
	size_t blocksK = round((double) K / (double) subK);
	size_t blocksN = round((double) N / (double) subN);


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

	/* STARTING MATRIX MULTIPLICATIONS! */

	size_t cnt, aInd, bInd, cInd;


	/* STRATEGY 1: Keeping same sublock of A in memory and exhausting matmuls paired with it */
	// for simplicity iterate keeping each of the "A blocks" in memory and pairing them will all the "B blocks"

	for (aInd = 0; aInd < blocksA; aInd++){
		load_subblock(subA, aInd, subM, subK, fpA, m, k);
		cnt = 0;
		bInd = aInd % blocksK;
		while (cnt < blocksN){
			// load b
			load_subblock(subB, bInd, subK, subN, fpB, k, n);
			
			/* Loading partial subC and adding this matmul to the partial result
			/* TODO: temp store this matmul and accumulate into C subblock later */
			cInd = (aInd / blocksK) * blocksN + cnt;
			load_subblock(subC, cInd, subM, subN, fpC, m, n);
			// do matmul (beta = 1.0 because adding to prior results in this strategy)
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
					subM, subN, subK, 1.0, subA, subK, subB, subN, 
					1.0, subC, subN);
			
			// save C
			save_subblock(subC, cInd, subM, subN, fpC, m, n);

			// get next B ind
			bInd += blocksK;

			// increment cnt
			cnt += 1;
		}
	}

	/* STRATEGY 2: Keeping same partial result block in memory and iterating over result blocks
	// iterate over each block in the output by keeping each "C block" in memory
	
	float beta;

	for (cInd = 0; cInd < blocksC; cInd++){
		// get starting inds for A row and B col
		aInd = (cInd / blocksN) * blocksK;
		bInd = (cInd % blocksN) * blocksK;

		cnt = 0;
		beta = 0;
		while (cnt < blocksK){
			// load subA
			load_subblock(subA, aInd, subM, subK, fpA, m, k);
			// load subB
			load_subblock(subB, bInd, subK, subN, fpB, k, n);

			// do matmul 
			// (beta = 0 for first iteration of inner loop then 1 because adding to prior results)
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
					subM, subN, subK, 1.0, subA, subK, subB, subN, 
					beta, subC, subN);

			// get next block in row of A
			aInd += 1;
			// get next block in col of B
			bInd += blocksK;

			// do next pairing of sub row/col blocks
			beta = 1;
			cnt += 1;
		}

		// save output subblock of C
		save_subblock(subC, cInd, subM, subN, fpC, m, n);

	}
	*/
	
	

	free(subA);
	free(subB);
	free(subC);

	fclose(fpA);
	fclose(fpB);
	fclose(fpC);






}