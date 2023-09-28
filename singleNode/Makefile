CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -O2 -Wall
PROFILE_FLAGS = -pg -no-pie -fno-builtin

all: saveMatrices singleNodeBigGEMM

saveMatrices: saveMatrices.c
	${CC} ${CFLAGS} -o $@ $^ -lm

singleNodeBigGEMM: singleNodeBigGEMM.c
	${CC} ${CFLAGS} -o $@ $^ -lm -lopenblas

clean:
	rm *.o