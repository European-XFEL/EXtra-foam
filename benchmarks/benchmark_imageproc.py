"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time

import numpy as np

from extra_foam.algorithms import (
    correct_image_data, mask_image_data, movingAvgImageData,
    nanmean_image_data
)


def _run_nanmean_image_array(data, data_type):
    data = data.astype(data_type)

    t0 = time.perf_counter()
    data_cpp = nanmean_image_data(data)
    dt_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    data_py = np.nanmean(data, axis=0)
    dt_py = time.perf_counter() - t0

    np.testing.assert_array_equal(data_cpp, data_py)

    # selected indices from an image array
    selected = [v for v in range(len(data)) if v not in [1, 11, 22]]

    t0 = time.perf_counter()
    data_cpp = nanmean_image_data(data, kept=selected)
    dt_cpp_sliced = time.perf_counter() - t0

    t0 = time.perf_counter()
    data_py = np.nanmean(data[selected], axis=0)
    dt_py_sliced = time.perf_counter() - t0

    np.testing.assert_array_equal(data_cpp, data_py)

    # two images
    t0 = time.perf_counter()
    data_cpp = nanmean_image_data((data[0, ...], data[1, ...]))
    dt_cpp_two = time.perf_counter() - t0

    t0 = time.perf_counter()
    data_py = np.nanmean(np.stack([data[0, ...], data[1, ...]]), axis=0)
    dt_py_two = time.perf_counter() - t0

    np.testing.assert_array_equal(data_cpp, data_py)

    print(f"\nnanmean_image_data with {data_type} - \n"
          f"dt (cpp para): {dt_cpp:.4f}, "
          f"dt (numpy): {dt_py:.4f}, \n"
          f"dt (cpp para) sliced: {dt_cpp_sliced:.4f}, "
          f"dt (numpy) sliced: {dt_py_sliced:.4f}, \n"
          f"dt (cpp para) two images: {dt_cpp_two:.4f}, "
          f"dt (numpy) two images: {dt_py_two:.4f}.")


def bench_nanmean_image_array(shape):
    data = np.random.rand(*shape)
    data[::2, ::2, ::2] = np.nan

    _run_nanmean_image_array(data, np.float32)
    _run_nanmean_image_array(data, np.float64)


def _run_moving_average_image_array(data, new_data, data_type):
    data = data.astype(data_type)
    new_data = new_data.astype(data_type)

    data_cpp = data.copy()
    t0 = time.perf_counter()
    movingAvgImageData(data_cpp, new_data, 5)
    dt_cpp = time.perf_counter() - t0

    data_py = data.copy()
    t0 = time.perf_counter()
    data_py += (new_data - data_py) / 5
    dt_py = time.perf_counter() - t0

    np.testing.assert_array_equal(data_cpp, data_py)

    print(f"\nmoving average with {data_type} - "
          f"dt (cpp para): {dt_cpp:.4f}, dt (numpy): {dt_py:.4f}")


def bench_moving_average_image_array(shape):
    data = np.random.rand(*shape)
    new_data = np.random.rand(*shape)

    _run_moving_average_image_array(data, new_data, np.float32)
    _run_moving_average_image_array(data, new_data, np.float64)


def _run_mask_image_array(data, mask, data_type, keep_nan=False):
    lb, ub = 0.2, 0.8
    data = data.astype(data_type)

    # mask nan
    data_cpp = data.copy()
    t0 = time.perf_counter()
    mask_image_data(data_cpp, keep_nan=keep_nan)
    dt_cpp_raw = time.perf_counter() - t0

    data_py = data.copy()
    t0 = time.perf_counter()
    if not keep_nan:
        data_py[np.isnan(data_py)] = 0
    dt_py_raw = time.perf_counter() - t0

    np.testing.assert_array_equal(data_cpp, data_py)

    # mask by threshold
    data_cpp = data.copy()
    t0 = time.perf_counter()
    mask_image_data(data_cpp, threshold_mask=(lb, ub), keep_nan=keep_nan)
    dt_cpp_th = time.perf_counter() - t0

    data_py = data.copy()
    t0 = time.perf_counter()
    if not keep_nan:
        data_py[np.isnan(data_py) | (data_py > ub) | (data_py < lb)] = 0
    else:
        data_py[(data_py > ub) | (data_py < lb)] = np.nan
    dt_py_th = time.perf_counter() - t0

    np.testing.assert_array_equal(data_cpp, data_py)

    # mask by both image and threshold
    data_cpp = data.copy()
    t0 = time.perf_counter()
    mask_image_data(data_cpp, image_mask=mask, threshold_mask=(lb, ub), keep_nan=keep_nan)
    dt_cpp = time.perf_counter() - t0

    data_py = data.copy()
    t0 = time.perf_counter()
    if not keep_nan:
        data_py[np.isnan(data_py) | mask | (data_py > ub) | (data_py < lb)] = 0
    else:
        data_py[mask | (data_py > ub) | (data_py < lb)] = np.nan
    dt_py = time.perf_counter() - t0

    np.testing.assert_array_equal(data_cpp, data_py)

    print(f"\nmask_image_data (keep_nan = {keep_nan}) with {data_type} - \n"
          f"dt (cpp para) raw: {dt_cpp_raw:.4f}, "
          f"dt (numpy) raw: {dt_py_raw:.4f}, \n"
          f"dt (cpp para) threshold: {dt_cpp_th:.4f}, "
          f"dt (numpy) threshold: {dt_py_th:.4f}, \n"
          f"dt (cpp para) threshold and image: {dt_cpp:.4f}, "
          f"dt (numpy) threshold and image: {dt_py:.4f}")


def bench_mask_image_array(shape):
    data = np.random.rand(*shape)
    data[::4, ::4, ::4] = np.nan
    mask = np.zeros(shape[-2:], dtype=np.bool)
    mask[::10, ::10] = True

    _run_mask_image_array(data, mask, np.float32, keep_nan=False)
    _run_mask_image_array(data, mask, np.float64, keep_nan=False)
    _run_mask_image_array(data, mask, np.float32, keep_nan=True)
    _run_mask_image_array(data, mask, np.float64, keep_nan=True)


def _run_correct_image_array(data, data_type, gain, offset):
    gain = gain.astype(data_type)
    offset = offset.astype(data_type)
    data = data.astype(data_type)

    # offset only
    data_cpp = data.copy()
    t0 = time.perf_counter()
    correct_image_data(data_cpp, offset=offset)
    dt_cpp_offset = time.perf_counter() - t0

    data_py = data.copy()
    t0 = time.perf_counter()
    data_py -= offset
    dt_py_offset = time.perf_counter() - t0

    np.testing.assert_array_almost_equal(data_cpp, data_py)

    # gain only
    data_cpp = data.copy()
    t0 = time.perf_counter()
    correct_image_data(data_cpp, gain=gain)
    dt_cpp_gain = time.perf_counter() - t0

    data_py = data.copy()
    t0 = time.perf_counter()
    data_py *= gain
    dt_py_gain = time.perf_counter() - t0

    np.testing.assert_array_almost_equal(data_cpp, data_py)

    # gain and offset
    data_cpp = data.copy()
    t0 = time.perf_counter()
    correct_image_data(data_cpp, gain=gain, offset=offset)
    dt_cpp_both = time.perf_counter() - t0

    data_py = data.copy()
    t0 = time.perf_counter()
    data_py = (data_py - offset) * gain
    dt_py_both = time.perf_counter() - t0

    np.testing.assert_array_almost_equal(data_cpp, data_py)

    print(f"\ncorrect_image_data (offset) with {data_type} - \n"
          f"dt (cpp para) offset: {dt_cpp_offset:.4f}, "
          f"dt (numpy) offset: {dt_py_offset:.4f}, \n"
          f"dt (cpp para) gain: {dt_cpp_gain:.4f}, "
          f"dt (numpy) gain: {dt_py_gain:.4f}, \n"
          f"dt (cpp para) gain and offset: {dt_cpp_both:.4f}, "
          f"dt (numpy) gain and offset: {dt_py_both:.4f}")


def bench_correct_gain_offset(shape):
    gain = np.random.randn(*shape)
    offset = np.random.randn(*shape)
    data = np.random.rand(*shape)

    _run_correct_image_array(data, np.float32, gain, offset)
    _run_correct_image_array(data, np.float64, gain, offset)


if __name__ == "__main__":
    print("*" * 80)
    print("Benchmark image processing")
    print("*" * 80)

    s = (32, 1096, 1120)

    with np.warnings.catch_warnings():
        np.warnings.simplefilter("ignore", category=RuntimeWarning)

        bench_nanmean_image_array(s)
        bench_moving_average_image_array(s)
        bench_mask_image_array(s)
        bench_correct_gain_offset(s)
