import ctypes
import numpy as np
import os
import sysconfig
import glob

# Locate the shared library dynamically
lib_dir = os.path.dirname(__file__)

# Find the shared library with the correct extension
lib_pattern = os.path.join(lib_dir, "align_ops.*.so")  # Matches align_ops.cpython-*.so
lib_candidates = glob.glob(lib_pattern)

if not lib_candidates:
    raise FileNotFoundError(f"Cannot find shared library in {lib_dir}")

lib_path = lib_candidates[0]  # Pick the first match (should be the only one)

# Load the shared library
align_ops = ctypes.CDLL(lib_path)

# Define function argument types
align_ops.align_sequences.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # log_probs
    ctypes.POINTER(ctypes.c_int64),  # targets
    ctypes.POINTER(ctypes.c_int64),  # paths (output)
    ctypes.POINTER(ctypes.c_float),  # scores (output)
    ctypes.c_int,  # batch_size
    ctypes.c_int,  # T
    ctypes.c_int,  # num_classes
    ctypes.c_int,  # L
    ctypes.c_int64  # blank
]

def align_sequences(log_probs: np.ndarray, targets: np.ndarray, blank: int):
    batch_size, T, num_classes = log_probs.shape
    L = targets.shape[1]
    # Ensure contiguous memory
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.ascontiguousarray(targets, dtype=np.int64)
    # Allocate memory for output arrays
    paths = np.zeros((batch_size, T), dtype=np.int64)
    scores = np.zeros(T, dtype=np.float32)
    # Call the C++ function
    align_ops.align_sequences(
        log_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        targets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        paths.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        batch_size,
        T,
        num_classes,
        L,
        blank
    )
    return paths, scores
