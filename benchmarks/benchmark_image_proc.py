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


def _run_nanmean_image_array(data_type):
    # image array
    data = np.ones((64, 1024, 512), dtype=data_type)
    data[::2, ::2, ::2] = np.nan

    t0 = time.perf_counter()
    nanmean_image_data(data)
    dt_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.nanmean(data, axis=0)
    dt_py = time.perf_counter() - t0

    # selected indices from an image array
    selected = [v for v in range(len(data)) if v not in [1, 11, 22]]

    t0 = time.perf_counter()
    nanmean_image_data(data, selected)
    dt_cpp_sliced = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.nanmean(data[selected], axis=0)
    dt_py_sliced = time.perf_counter() - t0

    # two images
    img = np.ones((1024, 512), dtype=data_type)
    img[::2, ::2] = np.nan

    t0 = time.perf_counter()
    nanmean_image_data(img, img)
    dt_cpp_two = time.perf_counter() - t0

    np.warnings.simplefilter("ignore", category=RuntimeWarning)
    t0 = time.perf_counter()
    np.nanmean([img, img], axis=0)
    dt_py_two = time.perf_counter() - t0

    print(f"\nnanmean_image_data with {data_type} - \n"
          f"dt (cpp para): {dt_cpp:.4f}, "
          f"dt (numpy): {dt_py:.4f}, \n"
          f"dt (cpp para) sliced: {dt_cpp_sliced:.4f}, "
          f"dt (numpy) sliced: {dt_py_sliced:.4f}, \n"
          f"dt (cpp para) two images: {dt_cpp_two:.4f}, "
          f"dt (numpy) two images: {dt_py_two:.4f}.")


def bench_nanmean_image_array():
    _run_nanmean_image_array(np.float32)
    _run_nanmean_image_array(np.float64)


def _run_moving_average_image_array(data_type):
    imgs = np.ones((64, 1024, 512), dtype=data_type)

    t0 = time.perf_counter()
    movingAvgImageData(imgs, imgs, 5)
    dt_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    imgs + (imgs - imgs) / 5
    dt_py = time.perf_counter() - t0

    print(f"\nmoving average with {data_type} - "
          f"dt (cpp para): {dt_cpp:.4f}, dt (numpy): {dt_py:.4f}")


def bench_moving_average_image_array():
    _run_moving_average_image_array(np.float32)
    _run_moving_average_image_array(np.float64)


def _run_mask_image_array(data_type):
    def prepare_data():
        dt = np.ones((64, 1024, 512), dtype=data_type)
        dt[::2, ::2, ::2] = np.nan
        return dt

    # mask nan
    data = prepare_data()
    t0 = time.perf_counter()
    mask_image_data(data)
    dt_cpp_raw = time.perf_counter() - t0

    data = prepare_data()
    t0 = time.perf_counter()
    data[np.isnan(data)] = 0
    dt_py_raw = time.perf_counter() - t0

    # mask by threshold
    data = prepare_data()
    t0 = time.perf_counter()
    mask_image_data(data, threshold_mask=(2., 3.))
    dt_cpp_th = time.perf_counter() - t0

    data = prepare_data()
    t0 = time.perf_counter()
    data[np.isnan(data)] = 0
    data[(data > 3) | (data < 2)] = 0
    dt_py_th = time.perf_counter() - t0

    # mask by both image and threshold
    mask = np.ones((1024, 512), dtype=np.bool)

    data = prepare_data()
    t0 = time.perf_counter()
    mask_image_data(data, image_mask=mask, threshold_mask=(2., 3.))
    dt_cpp = time.perf_counter() - t0

    data = prepare_data()
    t0 = time.perf_counter()
    data[np.isnan(data)] = 0
    data[:, mask] = 0
    data[(data > 3) | (data < 2)] = 0
    dt_py = time.perf_counter() - t0

    print(f"\nmask_image_data with {data_type} - \n"
          f"dt (cpp para) raw: {dt_cpp_raw:.4f}, "
          f"dt (numpy) raw: {dt_py_raw:.4f}, \n"
          f"dt (cpp para) threshold: {dt_cpp_th:.4f}, "
          f"dt (numpy) threshold: {dt_py_th:.4f}, \n"
          f"dt (cpp para) threshold and image: {dt_cpp:.4f}, "
          f"dt (numpy) threshold and image: {dt_py:.4f}")


def bench_mask_image_array():
    _run_mask_image_array(np.float32)
    _run_mask_image_array(np.float64)


def _run_correct_image_array(data_type, gain, offset):
    gain = gain.astype(data_type)
    offset = offset.astype(data_type)

    # offset only
    data = 2. * np.ones((64, 1024, 512), dtype=data_type)
    t0 = time.perf_counter()
    correct_image_data(data, offset=offset)
    dt_cpp_offset = time.perf_counter() - t0

    data = 2. * np.ones((64, 1024, 512), dtype=data_type)
    t0 = time.perf_counter()
    data -= offset
    dt_py_offset = time.perf_counter() - t0

    # gain only
    data = 2. * np.ones((64, 1024, 512), dtype=data_type)
    t0 = time.perf_counter()
    correct_image_data(data, gain=gain)
    dt_cpp_gain = time.perf_counter() - t0

    data = 2. * np.ones((64, 1024, 512), dtype=data_type)
    t0 = time.perf_counter()
    data *= gain
    dt_py_gain = time.perf_counter() - t0

    gain = gain.astype(data_type)
    offset = offset.astype(data_type)

    # gain and offset
    data = 2. * np.ones((64, 1024, 512), dtype=data_type)
    t0 = time.perf_counter()
    correct_image_data(data, gain=gain, offset=offset)
    dt_cpp_both = time.perf_counter() - t0

    data = 2. * np.ones((64, 1024, 512), dtype=data_type)
    t0 = time.perf_counter()
    data -= offset
    data *= gain
    dt_py_both = time.perf_counter() - t0

    print(f"\ncorrect_image_data (offset) with {data_type} - \n"
          f"dt (cpp para) offset: {dt_cpp_offset:.4f}, "
          f"dt (numpy) offset: {dt_py_offset:.4f}, \n"
          f"dt (cpp para) gain: {dt_cpp_gain:.4f}, "
          f"dt (numpy) gain: {dt_py_gain:.4f}, \n"
          f"dt (cpp para) gain and offset: {dt_cpp_both:.4f}, "
          f"dt (numpy) gain and offset: {dt_py_both:.4f}")


def bench_correct_gain_offset():
    gain = np.random.randn(64, 1024, 512)
    offset = np.random.randn(64, 1024, 512)
    _run_correct_image_array(np.float32, gain, offset)
    _run_correct_image_array(np.float64, gain, offset)


if __name__ == "__main__":
    bench_nanmean_image_array()
    bench_moving_average_image_array()
    bench_mask_image_array()
    bench_correct_gain_offset()
