'''
A utility file that includes helper functions for measuring memory usage and time.
Could be improved for more metrics
'''
import time
import tracemalloc
import matplotlib.pyplot as plt

def measure_memory(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Memory Usage - Current: {current / 10**6} MB; Peak: {peak / 10**6} MB")
        return result
    return wrapper

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution Time: {end - start:.4f} seconds")
        return result
    return wrapper
