# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gemm.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\ngemm.proto\x12\x04gemm\"B\n\x11MatMulInitRequest\x12\t\n\x01m\x18\x01 \x01(\x05\x12\t\n\x01k\x18\x02 \x01(\x05\x12\t\n\x01n\x18\x03 \x01(\x05\x12\x0c\n\x04pref\x18\x04 \x01(\t\"@\n\x12MatMulInitResponse\x12\x13\n\x0boperationId\x18\x01 \x01(\x05\x12\x15\n\restimatedTime\x18\x02 \x01(\x05\"\xa3\x01\n\x0bMatrixChunk\x12\x0c\n\x04rows\x18\x01 \x01(\x05\x12\x0c\n\x04\x63ols\x18\x02 \x01(\x05\x12\x10\n\x08rowStart\x18\x03 \x01(\x05\x12\x10\n\x08\x63olStart\x18\x04 \x01(\x05\x12\x13\n\x0boperationId\x18\x06 \x01(\x05\x12\x19\n\x11isLeftInputMatrix\x18\x07 \x01(\x08\x12\x16\n\x0eisOutputMatrix\x18\x08 \x01(\x08\x12\x0c\n\x04\x64\x61ta\x18\t \x03(\x02\x32\x88\x01\n\x06MatMul\x12\x42\n\rRequestMatMul\x12\x17.gemm.MatMulInitRequest\x1a\x18.gemm.MatMulInitResponse\x12:\n\x0eMatrixMultiply\x12\x11.gemm.MatrixChunk\x1a\x11.gemm.MatrixChunk(\x01\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'gemm_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_MATMULINITREQUEST']._serialized_start=20
  _globals['_MATMULINITREQUEST']._serialized_end=86
  _globals['_MATMULINITRESPONSE']._serialized_start=88
  _globals['_MATMULINITRESPONSE']._serialized_end=152
  _globals['_MATRIXCHUNK']._serialized_start=155
  _globals['_MATRIXCHUNK']._serialized_end=318
  _globals['_MATMUL']._serialized_start=321
  _globals['_MATMUL']._serialized_end=457
# @@protoc_insertion_point(module_scope)
