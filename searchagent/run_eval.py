"""
Measures various performance metrics:

- Latency: Measures latency for both GPU and CPU-bound tasks
- Precision: Assesses the quality of retrieved results
- Recall: Evaluates the completeness of retrieved results
- MRR (Mean Reciprocal Rank): Indicates how quickly the first relevant document is retrieved
- MAP (Mean Average Precision): Provides a comprehensive evaluation
by combining precision and the rank of relevant documents
"""

import time
from functools import wraps

import torch


def measure_latency_for_gpu(dummy_input, model, nb_iters=10):
    """Measure latency for GPU-bound tasks

    Args:
        dummy_input: A sample input matching your production input shape
        model: Model to be evaluated with dummy input for warmup
        nb_iters: Number of iterations to average the execution time
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # GPU warmup
            for _ in range(10):
                _ = model(dummy_input)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(nb_iters):
                result = func(*args, **kwargs)
            end.record()

            torch.cuda.current_stream().synchronize()
            avg_latency = start.elapsed_time(end) / nb_iters * 1e3

            # TODO: Add logger
            print(f"Average latency over {nb_iters} iterations: {avg_latency:.3f} ms")

            return result

        return wrapper

    return decorator


def measure_latency_for_cpu(func):
    """Measure latency for CPU-bound tasks"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        duration_ms = (end - start) * 1e3

        print(f"Duration: {duration_ms:.3f} ms")

        # TODO: Add custom logger
        return result

    return wrapper


def measure_ram(func):
    """Measure CPU memory"""

    @wraps(func)
    def wrapper(*args, **kwargs):

        result = func(*args, **kwargs)

        return result

    return wrapper


def measure_vram(func):
    """Measure GPU memory"""

    @wraps(func)
    def wrapper(*args, **kwargs):

        torch.cuda.reset_peak_memory_stats(device=None)

        result = func(*args, **kwargs)

        vram_used = torch.cuda.max_memory_allocated(device=None)

        print(f"{func.__name__} used {vram_used / (1024 ** 2):.2f} MB of VRAM")

        return result

    return wrapper


def calculate_precision():
    """Evaluate the relevance of retrieved documents to the user’s query"""
    pass


def calculate_recall():
    """Evaluate the proportion of relevant documents successfully retrieved by the system"""
    pass


def calculate_mean_reciprocal_rank():
    """Evaluate retrieval effectiveness based on the rank of the first relevant document"""
    pass


def calculate_mean_average_precision():
    """Evaluate the precision of retrieval across multiple queries"""
    pass
