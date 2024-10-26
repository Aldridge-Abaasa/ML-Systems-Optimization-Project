# Handles data loading(need more industry relevant datatypes) and preprocessing (preprocessing appoaches could be investigated further)

import numpy as np

def load_data():
    # Dummy synthetic data for demonstration
    X = np.random.randn(1000, 20).astype(np.float32)
    y = np.random.randn(1000, 5).astype(np.float32)
    return X, y
