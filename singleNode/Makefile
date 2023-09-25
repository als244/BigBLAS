CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -O2 -Wall
PROFILE_FLAGS = -pg -no-pie -fno-builtin

all: cpu_gemm_openblas

cpu_gemm_openblas.o: cpu_gemm_openblas.c
	gcc -I/usr/include/OpenBLAS/include -c -O2 cpu_gemm_openblas.c 

cpu_gemm_openblas: cpu_gemm_openblas.o
	gcc -L/usr/include/OpenBLAS/libopenblas.a -o cpu_gemm_openblas cpu_gemm_openblas.o -lm -lopenblas

clean:
	rm *.o