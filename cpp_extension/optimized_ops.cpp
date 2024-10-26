// optimized_ops.cpp  (optimized implementations of matrix multiplication and ReLU operations.)
#include <iostream>
#include <vector>
#include <omp.h>

// Matrix multiplication using OpenMP for parallelism
extern "C" void matrix_multiply(float* A, float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ReLU function using vectorization
extern "C" void relu(float* X, int size) {
    #pragma omp simd
    for (int i = 0; i < size; ++i) {
        X[i] = std::max(0.0f, X[i]);
    }
}
