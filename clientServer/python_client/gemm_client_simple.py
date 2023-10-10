import sys
import logging
import random
import numpy as np

import grpc
import gemm_simple_pb2
import gemm_simple_pb2_grpc


def createMatricesInputMessage(operationId, A, B):
	m = A.shape[0]
	k = A.shape[1]
	n = B.shape[1]

	matrixData = np.ravel(A).tolist() + np.ravel(B).tolist()

	matricesInputMessage = gemm_simple_pb2.MatricesInput(
		operationId = operationId,
		m = m,
		k = k,
		n = n
	)

	matricesInputMessage.data[:] = matrixData
	return matricesInputMessage

def do_rpc_matmul(stub, A, B):

	m = A.shape[0]
	k = A.shape[1]
	n = B.shape[1]

	initMatMulRequest = gemm_simple_pb2.MatMulInitRequest(m=m, k=k, n=n)
	initMatMulResponse = stub.RequestMatMul(initMatMulRequest)

	operationId = initMatMulResponse.operationId

	matmulRpcInput = createMatricesInputMessage(operationId, A, B)

	output = stub.MatrixMultiply(matmulRpcInput)

	return output


def run():

	# DO IT WITH KNOWN INPUTS

	data_dir = "/home/shein/Documents/grad_school/research/BigBLAS/data/"
	A_filename, B_filename = "A_8192_4096", "B_4096_16384"
	A = np.reshape(np.fromfile(data_dir + A_filename, dtype=np.float32), (8192, 4096))
	B = np.reshape(np.fromfile(data_dir + B_filename, dtype=np.float32), (4096, 16384))

	# NOTE(gRPC Python Team): .close() is possible on a channel and should be
	# used in circumstances in which the with statement does not fit the needs
	# of the code.
	
	with grpc.insecure_channel("localhost:50051") as channel:
		stub = gemm_simple_pb2_grpc.MatMulStub(channel)
		matrixOutput = do_rpc_matmul(stub, A, B)	
		
		operationId = matrixOutput.operationId
		print("Finished Operation with ID = " + str(operationId) + "\n")
		m, n = matrixOutput.m, matrixOutput.n
		print("Output Dims: (" + str(m) + ", " + str(n) + ")\n")
		matrixData = matrixOutput.data
		C = np.reshape(np.asarray(matrixData, dtype=np.float32), (m, n))


if __name__ == "__main__":
	logging.basicConfig()
	run()