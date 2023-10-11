import sys
import logging
import math
import time
import numpy as np

import grpc
import mathcloud.rpc.pyproto.mathcloud_pb2
import mathcloud.rpc.pyproto.mathcloud_pb2_grpc

class Matrix:

	def __init__(self):
		pass

	def __add__(self, other):

		if type(other) is Matrix:

			## do matmul

		else:
			raise TypeMismatchError("Cannot add <Matrix> with " + type(other))

	def __mul__(self, other):

		if type(other) is Matrix:

			## get estimation to determine if call to gRPC
			## for now assume always call
			return 

		elif type(other) is Vector:

			## do matrixVec

		elif other.isnumeric():

			## do scalar

		else:
			raise TypeMismatchError("Cannot multiply <Matrix> with " + type(other))
