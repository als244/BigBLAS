CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -O2 -Wall
PROFILE_FLAGS = -pg -no-pie -fno-builtin

all: saveMatrices singleNodeBigGEMM openblas_gemm cublas_gemm

saveMatrices: saveMatrices.c
	${CC} ${CFLAGS} -o $@ $^ -lm

singleNodeBigGEMM: singleNodeBigGEMM.c
	nvcc -O3 -o $@ $^ -lm -lopenblas -lcublasLt

openblas_gemm: openblas_gemm.c
	${CC} ${CFLAGS} -o $@ $^ -lm -lopenblas

cublas_gemm: cublas_gemm.cu
	nvcc -O3 cublas_gemm.cu -o cublas_gemm -lcublasLt -lcurand

clean:
	rm *.o