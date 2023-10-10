import logging
import math
import time
import numpy as np

import grpc
import gemm_pb2
import gemm_pb2_grpc



def getEstimatedTime(operation):
	## TODO: build fancier function here
	return 0

def createInitMatMulResponse(operation):
	
	## later on do function for this
	estimatedTime = getEstimatedTime(operation)
	response = gemm_pb2.MatMulInitResponse(
		operationId = operation.id,
		estimatedTime = estimatedTime
	)
	return response

def createMatrixChunkOutput(col_start, row_start, operation_id, out_matrix):
	shape = np.shape(out_matrix)
	matrixData = np.ravel(out_matrix).tolist()
	output_chunk = gemm_pb2.MatrixChunk(
		rows=shape[0],
		cols=shape[1],
		rowStart = row_start,
		colStart = col_start,
		operationId = operation_id,
		isLeftInputMatrix = False,
		isOutputMatrix = True,
		data = matrixData
	)
	return output_chunk

class OperationProgress():

	def __init__(self, operation_id, blocksM, blocksK, blocksN):
		self.id = operation_id
		self.blocksM = blocksM
		self.blocksK = blocksK
		self.blocksN = blocksN
		self.resultDepends, self.dependsOnA, self.dependsOnB = createInitDependencies(blocksM, blocksK, blocksN)
		self.completedBlocks = set()
		self.blocksAReceived = set()
		self.blocksBReceived = set()
		self.blocksAInMem = set()
		self.blcoksBinMem = set()

	# returns:
	# 	dictionary of resultBlockId -> set of (blockAId, blockBId) tuples that need to be matMul'd and added
	# 	dict of aBlockId -> set of result blocks it is dependency for
	#	dict of bBlockId -> set of result blocks it is dependency for
	def createInitDependencies(blocksM, blocksK, blocksN):
		remaining = {}
		dependsOnA, dependsOnB = {}, {}
		blocksC = blocksM * blocksN
		for cInd in range(blocksC):
			remaining[cInd] = set()
			aInd = (cInd // blocksN) * blocksK;
			bInd = (cInd % blocksN) * blocksK;
			cnt = 0
			while (cnt < blocksK):
				remaining[cInd].add((aInd, bInd))
				if aInd not in dependsOnA:
					dependsOnA[aInd] = set()
				if bInd not in dependsOnB:
					dependsOnB[bInd] = set()
				dependsOnA[aInd].add(cInd)
				dependsOnB[bInd].add(cInd)
				aInd += 1
				bInd += blocksK
				cnt += 1
		return remaining, dependsOnA, dependsOnB
		

class Operation():

	def __init__(self, operation_id, m, k, n, pref):
		self.id = operation_id
		self.M = m
		self.K = k
		self.N = n
		self.pref = pref
		self.subM, self.subK, self.subN = getPartitions(m, k, n, pref)
		self.blocksM = math.ceil(m / subM)
		self.blocksK = math.ceil(k / subK)
		self.blocksN = math.ceil(n / subN)
		self.progress = OperationProgress(operation_id, blocksM, blocksK, blocksN)


	def getPartitions(m, k, n, pref):
		## TODO: build fancier function here
		return 32768, 65536, 32768

	def getBlockId(self, row, col, isLeft, isOutput):
		if isLeft:
			subRow = row // self.subM
			subCol = col // self.subK
			return subRow * blocksK + subCol
		elif not isOutput:
			subRow = row // self.subK
			subCol = col // self.subN
			return subRow * blocksN + subCol
		else:
			subRow = row // self.subM
			subCol = row // self.subN
			return subRow * blocksN + subCol


		

class MatMulServicer(gemm_pb2_grpc.MatMulServicer):
    """Provides methods that implement functionality of route guide server."""

    def __init__(self):
    	self.operationCounter = 0
        self.operations = {}


    def RequestMatMul(self, request, context):
    	M, K, N = request.m, request.k, request.n
    	pref = request.pref
    	# make sure this is set atomic
    	operation_id = self.operationCounter
    	self.operationCounter += 1
    	newOperation = Operation(operation_id, M, K, N, pref)
    	self.operations[operation_id] = newOperation

    	initResponse = createInitMatMulResponse(newOperation)
    	return initResponse


    def MatrixMultiply(self, request_iterator, context):

    	for matrix_chunk in request_iterator:

    		## see if the chunk is valid
    		if matrix_chunk.operationId not in self.operations:
    			# ERROR! 
    			print("Operation not underway")
    			continue

    		## determine what block(s) matrix incoming data stream refers to
    		## save data in server/cluster filesystem

    		# A
    		if  matrix_chunk.isLeftInputMatrix:

    		# B
    		else:



    		## see if necessary blocks have been sent over 
    		## in order to compute series of matmuls and adds
    		## to then send back a result block


    		result = np.matmul()
    		outputChunk = createMatrixChunkOutput(result) 
    		yield outputChunk
        


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gemm_pb2_grpc.add_MatMulServicer_to_server(
        MatMulServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()