import time
import pytest

import numpy as np

from extra_foam.algorithms import (
    nanmean, nansum
)


def benchmark_nan_without_axis(f_cpp, f_py, shape, dtype):
    data = np.random.randn(*shape).astype(dtype) + 1.  # shift to avoid very small mean
    data[:, :3, ::3] = np.nan

    t0 = time.perf_counter()
    ret_cpp = f_cpp(data)
    dt_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    ret_py = f_py(data)
    dt_py = time.perf_counter() - t0

    assert ret_cpp == pytest.approx(ret_py, rel=1e-4)

    print(f"\nwithout axis, dtype = {dtype} - \n"
          f"dt (cpp): {dt_cpp:.4f}, "
          f"dt (numpy): {dt_py:.4f}")


def benchmark_nan_keep_zero_axis(f_cpp, f_py, shape, dtype):
    data = np.random.randn(*shape).astype(dtype=dtype) + 1.  # shift to avoid very small mean
    data[:, :3, ::3] = np.nan

    t0 = time.perf_counter()
    ret_cpp = f_cpp(data, axis=(-2, -1))
    dt_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    ret_py = f_py(data, axis=(-2, -1))
    dt_py = time.perf_counter() - t0

    assert ret_cpp == pytest.approx(ret_py, rel=1e-4)

    print(f"\nkeep zero axis, dtype = {dtype} - \n"
          f"dt (cpp): {dt_cpp:.4f}, "
          f"dt (numpy): {dt_py:.4f}")


if __name__ == "__main__":
    print("*" * 80)
    print("Benchmark statistics functions")
    print("*" * 80)

    s = (16, 1096, 1120)

    for f_cpp, f_py in [(nansum, np.nansum), (nanmean, np.nanmean)]:
        print(f"\n----- {f_cpp.__name__} ------")
        benchmark_nan_without_axis(f_cpp, f_py, s, np.float32)
        benchmark_nan_without_axis(f_cpp, f_py, s, np.float64)
        benchmark_nan_keep_zero_axis(f_cpp, f_py, s, np.float32)
        benchmark_nan_keep_zero_axis(f_cpp, f_py, s, np.float64)
