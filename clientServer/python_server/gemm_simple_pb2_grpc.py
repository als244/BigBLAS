# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import gemm_simple_pb2 as gemm__simple__pb2


class MatMulStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RequestMatMul = channel.unary_unary(
                '/gemm_simple.MatMul/RequestMatMul',
                request_serializer=gemm__simple__pb2.MatMulInitRequest.SerializeToString,
                response_deserializer=gemm__simple__pb2.MatMulInitResponse.FromString,
                )
        self.MatrixMultiply = channel.unary_unary(
                '/gemm_simple.MatMul/MatrixMultiply',
                request_serializer=gemm__simple__pb2.MatricesInput.SerializeToString,
                response_deserializer=gemm__simple__pb2.MatrixOutput.FromString,
                )


class MatMulServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RequestMatMul(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MatrixMultiply(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MatMulServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RequestMatMul': grpc.unary_unary_rpc_method_handler(
                    servicer.RequestMatMul,
                    request_deserializer=gemm__simple__pb2.MatMulInitRequest.FromString,
                    response_serializer=gemm__simple__pb2.MatMulInitResponse.SerializeToString,
            ),
            'MatrixMultiply': grpc.unary_unary_rpc_method_handler(
                    servicer.MatrixMultiply,
                    request_deserializer=gemm__simple__pb2.MatricesInput.FromString,
                    response_serializer=gemm__simple__pb2.MatrixOutput.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'gemm_simple.MatMul', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MatMul(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RequestMatMul(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gemm_simple.MatMul/RequestMatMul',
            gemm__simple__pb2.MatMulInitRequest.SerializeToString,
            gemm__simple__pb2.MatMulInitResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MatrixMultiply(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gemm_simple.MatMul/MatrixMultiply',
            gemm__simple__pb2.MatricesInput.SerializeToString,
            gemm__simple__pb2.MatrixOutput.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)