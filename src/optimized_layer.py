# This file defines neural network layers, leveraging the C++ operations through Python’s ctypes or cffi for integration.

import numpy as np
import ctypes
import os

# Load C++ optimized library
optimized_lib = ctypes.CDLL(os.path.join("cpp_extension", "build", "liboptimized_ops.so"))

class OptimizedLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim).astype(np.float32)
        self.bias = np.zeros(output_dim, dtype=np.float32)

    def forward(self, X):
        output = np.empty((X.shape[0], self.weights.shape[1]), dtype=np.float32)
        optimized_lib.matrix_multiply(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(X.shape[0]), ctypes.c_int(self.weights.shape[1]), ctypes.c_int(self.weights.shape[0])
        )
        return output + self.bias
