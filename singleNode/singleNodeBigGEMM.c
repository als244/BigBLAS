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
void load_subblock(float * subBlock, size_t blockInd, size_t subRows, size_t subCols, FILE * fpBigMatrix, size_t rows, size_t cols){

}

void save_subblock(float * subBlock, size_t blockInd, size_t subRows, size_t subCols, FILE * fpBigMatrix, size_t rows, size_t cols){

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
	fread(&A_metadata, sizeof(struct matrixMeta), 1, fpAMeta);
	fclose(fpAMeta);

	// get B metadata
	fpBMeta = fopen(pathBMeta, "r");
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
	int A_type = A_metadata.typeID;
	int B_type = B_metadata.typeID;

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
	size_t blocksK = round((double) M / (double) subM);
	size_t blocksN = round((double) M / (double) subM);


	size_t blocksA = blocksM * blocksK;
	size_t blocksB = blocksK * blocksN;

	/* STRATEGY 1: Keeping same sublock of A in memory and exhausting matmuls paired with it */
	// for simplicity iterate keeping each of the "A blocks" in memory and pairing them will all the "B blocks"

	size_t cnt, bInd, cInd;
	for (size_t aInd = 0; aInd < blocksA; aInd++){
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

			// do matmul

			// save C
			save_subblock(subC, cInd, subM, subN, fpC, m, n);

			// get next B ind
			bInd += blocksK;

			// increment cnt
			cnt += 1;
		}
	} 



	/* STRATEGY 2: Keeping same partial result block in memory and iterating over result blocks */






}