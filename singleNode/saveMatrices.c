#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define M_PI 3.14159265358979323846

// assume typeIDs: float -> 0, int -> 1, long -> 2, uint -> 3
struct matrixMeta {
	size_t rows;
	size_t cols;
	int typeID;
};

void sampleNormal(float * X, size_t size, float mean, float var) {
	float x, y, z, std, val;
	for (size_t i = 0; i < size; i++){
		x = (float)rand() / RAND_MAX;
		y = (float)rand() / RAND_MAX;
		z = sqrtf(-2 * logf(x)) * cosf(2 * M_PI * y);
		std = sqrtf(var);
		val = std * z + mean;
		X[i] = val;
	}
}

void populateIndices(float * X, size_t size, size_t startInd){
	for (size_t i = startInd; i < startInd + size; i++){
		X[i] = (float) i;
	}
}

int main(int argc, char * argv[]){


	size_t m = atol(argv[1]);
	size_t k = atol(argv[2]);
	size_t n = atol(argv[3]);

	// deal with various types later, but include as input...
	int typeID = atoi(argv[4]);

	char * fillType = argv[5];

	char * fileNameA = argv[6];
	char * fileNameB = argv[7];

	char * pathA, *pathAMeta, *pathB, *pathBMeta;
	
	asprintf(&pathA, "/home/shein/Documents/grad_school/research/BigBLAS/data/%s", fileNameA);
	asprintf(&pathAMeta, "/home/shein/Documents/grad_school/research/BigBLAS/data/%s.metadata", fileNameA);
	asprintf(&pathB, "/home/shein/Documents/grad_school/research/BigBLAS/data/%s", fileNameB);
	asprintf(&pathBMeta, "/home/shein/Documents/grad_school/research/BigBLAS/data/%s.metadata", fileNameB);

	FILE * fpA, *fpAMeta, * fpB, * fpBMeta;


	// dealing with A matrix
	printf("Writing to matrix A!\n\n");

	// write metadata struct
	struct matrixMeta A_metadata = {m, k, typeID};
	fpAMeta = fopen(pathAMeta, "w+");
	free(pathAMeta);
	fwrite(&A_metadata, sizeof(struct matrixMeta), 1, fpAMeta);
	fclose(fpAMeta);

	// build matrix A and write in chunks
	size_t chunk_size = 4000000000;
	float * A_chunk = (float *) calloc(chunk_size, sizeof(float));
	fpA = fopen(pathA, "w+");
	free(pathA);

	size_t remain = m * k;
	size_t chunk_start = 0;
	while (remain > 0){
		if (chunk_size > remain){
			chunk_size = remain;
		}
		if (strcmp(fillType, "random") == 0) {
			sampleNormal(A_chunk, chunk_size, 0, 1);
		}
		else{
			populateIndices(A_chunk, chunk_size, chunk_start);
			chunk_start += chunk_size;
		}
		fwrite(A_chunk, sizeof(float), chunk_size, fpA);
		remain -= chunk_size;
	}

	free(A_chunk);
	fclose(fpA);

	// dealing with B matrix
	printf("Writing to matrix B!\n\n");

	// write metadata struct
	struct matrixMeta B_metadata = {k, n, typeID};
	fpBMeta = fopen(pathBMeta, "w+");
	free(pathBMeta);
	fwrite(&B_metadata, sizeof(struct matrixMeta), 1, fpBMeta);
	fclose(fpBMeta);

	// build matrix B and write in chunks
	chunk_size = 4000000000;
	float * B_chunk = (float *) calloc(chunk_size, sizeof(float));
	fpB = fopen(pathB, "w+");
	free(pathB);

	remain = k * n;
	chunk_start = 0;
	while (remain > 0){
		if (chunk_size > remain){
			chunk_size = remain;
		}
		if (strcmp(fillType, "random") == 0) {
			sampleNormal(B_chunk, chunk_size, 0, 1);
		}
		else{
			populateIndices(A_chunk, chunk_size, chunk_start);
			chunk_start += chunk_size;
		}
		fwrite(B_chunk, sizeof(float), chunk_size, fpB);
		remain -= chunk_size;
	}

	free(B_chunk);
	fclose(fpB);

	return 0;
}