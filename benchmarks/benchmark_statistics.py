import time
import pytest

import numpy as np

from extra_foam.algorithms import (
    nanmean, nansum
)


def benchmark_nan(f_cpp, f_py, shape):
    data = np.random.randn(*shape)
    data[::2, ::2] = np.nan

    t0 = time.perf_counter()
    for i in range(100):
        ret_cpp = f_cpp(data)
    dt_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(100):
        ret_py = f_py(data)
    dt_py = time.perf_counter() - t0

    assert ret_cpp == pytest.approx(ret_py)

    print(f"\n{f_cpp.__name__} with {float} - \n"
          f"dt (cpp): {dt_cpp:.4f}, "
          f"dt (numpy): {dt_py:.4f}")


if __name__ == "__main__":
    print("*" * 80)
    print("Benchmark image processing")
    print("*" * 80)

    s = (1096, 1120)

    for f_cpp, f_py in [(nanmean, np.nanmean),
                        (nansum, np.nansum)]:
        benchmark_nan(f_cpp, f_py, s)
