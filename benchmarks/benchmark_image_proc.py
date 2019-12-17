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
    nanmeanImageArray, nanmeanTwoImages, movingAverageImageArray,
    mask_image_array, subDarkImageArray,
)


def _run_nanmean_image_array(data_type):
    data = np.ones((64, 1024, 512), dtype=data_type)
    data[::2, ::2, ::2] = np.nan

    t0 = time.perf_counter()
    nanmeanImageArray(data)
    dt_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.nanmean(data, axis=0)
    dt_py = time.perf_counter() - t0

    selected = [v for v in range(len(data)) if v not in [1, 11, 22]]

    t0 = time.perf_counter()
    nanmeanImageArray(data, selected)
    dt_cpp_sliced = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.nanmean(data[selected], axis=0)
    dt_py_sliced = time.perf_counter() - t0

    print(f"\nnanmeanImageArray with {data_type} - "
          f"dt (cpp para): {dt_cpp:.4f}, dt (numpy): {dt_py:.4f}, "
          f"dt (cpp para sliced): {dt_cpp_sliced:.4f}, dt (numpy sliced): {dt_py_sliced:.4f}")


def bench_nanmean_image_array():
    _run_nanmean_image_array(np.float32)
    _run_nanmean_image_array(np.float64)


def _run_nanmean_two_images(data_type):
    img = np.ones((1024, 512), dtype=data_type)
    img[::2, ::2] = np.nan

    t0 = time.perf_counter()
    nanmeanTwoImages(img, img)
    dt_cpp = time.perf_counter() - t0

    imgs = np.ones((2, 1024, 512), dtype=data_type)
    imgs[:, ::2, ::2] = np.nan

    t0 = time.perf_counter()
    nanmeanImageArray(imgs)
    dt_cpp_2 = time.perf_counter() - t0

    print(f"\nnanmeanTwoImages with {data_type} - "
          f"dt (cpp para): {dt_cpp:.4f}, dt (cpp para2): {dt_cpp_2:.4f}")


def bench_nanmean_with_two_images():
    _run_nanmean_two_images(np.float32)
    _run_nanmean_two_images(np.float64)


def _run_moving_average_image_array(data_type):
    imgs = np.ones((64, 1024, 512), dtype=data_type)

    t0 = time.perf_counter()
    movingAverageImageArray(imgs, imgs, 5)
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
    # mask by threshold
    data = np.ones((64, 1024, 512), dtype=data_type)
    t0 = time.perf_counter()
    mask_image_array(data, threshold_mask=(2., 3.))  # every elements are masked
    dt_cpp_th = time.perf_counter() - t0

    data = np.ones((64, 1024, 512), dtype=data_type)
    t0 = time.perf_counter()
    data[(data > 3) | (data < 2)] = 0
    dt_py_th = time.perf_counter() - t0

    # mask by image
    mask = np.ones((1024, 512), dtype=np.bool)

    data = np.ones((64, 1024, 512), dtype=data_type)
    t0 = time.perf_counter()
    mask_image_array(data, image_mask=mask)
    dt_cpp = time.perf_counter() - t0

    data = np.ones((64, 1024, 512), dtype=data_type)
    t0 = time.perf_counter()
    data[:, mask] = 0
    dt_py = time.perf_counter() - t0

    print(f"\nmaskImageArray with {data_type} - "
          f"dt (cpp para) threshold: {dt_cpp_th:.4f}, "
          f"dt (numpy) threshold: {dt_py_th:.4f}, "
          f"dt (cpp para) image: {dt_cpp:.4f}, dt (numpy) image: {dt_py:.4f}")


def bench_mask_image_array():
    _run_mask_image_array(np.float32)
    _run_mask_image_array(np.float64)


def _run_nan2zero_image_array(data_type):
    # mask by threshold
    data = np.ones((64, 1024, 512), dtype=data_type)
    data[::2, ::2, ::2] = np.nan

    t0 = time.perf_counter()
    mask_image_array(data)
    dt_cpp = time.perf_counter() - t0

    # need a fresh data since number of nans determines the performance
    data = np.ones((64, 1024, 512), dtype=data_type)
    data[::2, ::2, ::2] = np.nan

    t0 = time.perf_counter()
    data[np.isnan(data)] = 0
    dt_py = time.perf_counter() - t0

    print(f"\nnanToZeroImageArray with {data_type} - "
          f"dt (cpp para): {dt_cpp:.4f}, dt (numpy): {dt_py:.4f}")


def bench_nan2zero_image_array():
    _run_nan2zero_image_array(np.float32)
    _run_nan2zero_image_array(np.float64)


def _run_sub_dark_image_array(data_type):
    # mask by threshold
    data = np.ones((64, 1024, 512), dtype=data_type)
    dark = 0.5 * np.ones((64, 1024, 512), dtype=data_type)
    t0 = time.perf_counter()
    subDarkImageArray(data, dark)
    dt_cpp = time.perf_counter() - t0

    # need a fresh data since number of nans determines the performance
    data = np.ones((64, 1024, 512), dtype=data_type)
    dark = 0.5 * np.ones((64, 1024, 512), dtype=data_type)

    t0 = time.perf_counter()
    data -= dark
    dt_py = time.perf_counter() - t0

    print(f"\nsubDarkImageArray with {data_type} - "
          f"dt (cpp para): {dt_cpp:.4f}, dt (numpy): {dt_py:.4f}")


def bench_sub_dark_image_array():
    _run_sub_dark_image_array(np.float32)
    _run_sub_dark_image_array(np.float64)


if __name__ == "__main__":
    bench_nanmean_image_array()
    bench_nanmean_with_two_images()
    bench_moving_average_image_array()
    bench_mask_image_array()
    bench_nan2zero_image_array()
    bench_sub_dark_image_array()
