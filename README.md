# ML-Systems-Optimization-Project
This project demonstrates foundational machine learning system optimization, focusing on neural network layer efficiency, memory management, and hardware-aware optimization using C++ and Python integration. By integrating C++ for performance-critical operations, this project accelerates model training and inference, tracking improvements in runtime and memory usage.

# Features:
- C++ Optimization for Critical Operations: Accelerated matrix multiplication and ReLU operations, demonstrating cache optimization and parallelism.
- Performance Monitoring: Metrics for training time, memory usage, and operation profiling across layers.
- Python/C++ Integration: C++ extensions for enhanced performance in Python workflows.
- Layered Neural Network Implementation: Python neural network layers with support for both C++-optimized and Python-based execution.
- Comprehensive Memory & Profiling Utilities: Tracks memory footprint and provides layer-wise performance metrics.

# Key Implementations
- Optimized Matrix Multiplication: Cache-friendly, parallelized matrix multiplication using OpenMP.
- ReLU Activation Optimization: Vectorized ReLU function for faster gradient calculation.
- Performance Profiling: Built-in utilities to track memory usage, forward/backward pass time, and layer-specific performance.

## Results

Metric	Value
Memory Usage	4.2 MB
Training Time	2.5 seconds

## Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt


2. **Build C++ Extension**:
 ```bash cd cpp_extension
mkdir build && cd build
cmake ..
make



3. **Usage**:
 ```bash from src.data_loader import load_data
from src.performance_nn import PerformanceOptimizedNN

# Load data
X, y = load_data()

# Initialize and train model
model = PerformanceOptimizedNN([20, 15, 10, 5])
model.train(X, y, epochs=10, learning_rate=0.01)


