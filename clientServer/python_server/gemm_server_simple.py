from concurrent import futures

import logging
import math
import time
import numpy as np

import grpc
import gemm_simple_pb2
import gemm_simple_pb2_grpc


def createInitMatMulResponse(operation):
	
	## later on do function for estimating time
	response = gemm_simple_pb2.MatMulInitResponse(
		operationId = operation.id
	)
	return response

def createMatrixOutput(operation_id, out_matrix):
	shape = np.shape(out_matrix)
	matrixData = np.ravel(out_matrix).tolist()
	outputMatrix = gemm_simple_pb2.MatrixChunk(
		operationId=operation_id,
		m=shape[0],
		n=shape[1]
	)

	outputMatrix.data[:] = matrixData
	return outputMatrix

class Operation():

	def __init__(self, operation_id, m, k, n, pref):
		self.id = operation_id
		self.M = m
		self.K = k
		self.N = n
		

class MatMulServicer(gemm_simple_pb2_grpc.MatMulServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self):
    	self.operationCounter = 0
        self.operations = {}


    def RequestMatMul(self, request, context):
    	M, K, N = request.m, request.k, request.n
    	# make sure this is set atomic
    	operation_id = self.operationCounter
    	self.operationCounter += 1
    	newOperation = Operation(operation_id, M, K, N, pref)
    	self.operations[operation_id] = newOperation

    	initResponse = createInitMatMulResponse(newOperation)
    	return initResponse


    def MatrixMultiply(self, request, context):

    	operationId = request.operationId
    	m, k, n = request.m, request.k, request.n

    	if request.operationId not in self.operations:
    		#error
    		print("Operation not granted!")
    		return

    	operation = self.operations[operationId]

    	if m != operation.M or k != operation.K or n != operation.N:
    		#error
    		print("Operation dims do not match!")
    		return

    	data = request.data

    	A = np.reshape(np.asarray(data[:m * k], dtype=np.float32), (m, k))
    	B = np.reshape(np.asarray(data[m * k: ], dtype=np.float32), (k, n))

    	C = np.matmul(A, B)

    	outputMatrix = createMatrixOutput(operationId, C)
    	return outputMatrix



        


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gemm_simple_pb2_grpc.add_MatMulServicer_to_server(
        MatMulServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()