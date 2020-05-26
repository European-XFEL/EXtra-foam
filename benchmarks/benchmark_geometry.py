"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp
import time

import numpy as np

from extra_foam.config import config

_IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']
_RAW_IMAGE_DTYPE = config['SOURCE_RAW_IMAGE_DTYPE']

_data_sources = [(_RAW_IMAGE_DTYPE, 'raw'), (_IMAGE_DTYPE, 'calibrated')]

_geom_path = osp.join(osp.dirname(osp.abspath(__file__)), "../extra_foam/geometries")


def _benchmark_1m_imp(geom_fast_cls, geom_cls, geom_file, quad_positions=None):

    for from_dtype, from_str in _data_sources:
        n_pulses = 64
        modules = np.ones((n_pulses,
                           geom_fast_cls.n_modules,
                           geom_fast_cls.module_shape[0],
                           geom_fast_cls.module_shape[1]), dtype=from_dtype)

        # assemble with geometry and quad position in EXtra-geom

        if quad_positions is not None:
            geom = geom_cls.from_h5_file_and_quad_positions(geom_file, quad_positions)
        else:
            geom = geom_cls.from_crystfel_geom(geom_file)
        assembled = geom.output_array_for_position_fast((n_pulses,), dtype=_IMAGE_DTYPE)
        t0 = time.perf_counter()
        geom.position_all_modules(modules, out=assembled)
        dt_geom = time.perf_counter() - t0

        # stack only

        geom = geom_fast_cls()
        assembled = np.full((n_pulses, *geom.assembledShape()), np.nan, dtype=_IMAGE_DTYPE)
        t0 = time.perf_counter()
        geom.position_all_modules(modules, assembled)
        dt_foam_stack = time.perf_counter() - t0

        # assemble with geometry and quad position

        if quad_positions is not None:
            geom = geom_fast_cls.from_h5_file_and_quad_positions(geom_file, quad_positions)
        else:
            geom = geom_fast_cls.from_crystfel_geom(geom_file)
        assembled = np.full((n_pulses, *geom.assembledShape()), np.nan, dtype=_IMAGE_DTYPE)
        t0 = time.perf_counter()
        geom.position_all_modules(modules, assembled)
        dt_foam = time.perf_counter() - t0

        print(f"\nposition all modules for {geom_cls.__name__} (from {from_str} data) - \n"
              f"  dt (foam stack only): {dt_foam_stack:.4f}, dt (foam): {dt_foam:.4f}, "
              f"dt (geom): {dt_geom:.4f}")

        if modules.dtype == _IMAGE_DTYPE:
            t0 = time.perf_counter()
            geom.dismantle_all_modules(assembled, modules)
            dt_foam_dismantle = time.perf_counter() - t0

            print(f"\ndismantle all modules for {geom_cls.__name__} (from {from_str} data) - \n"
                  f"  dt (foam): {dt_foam_dismantle:.4f}")


def benchmark_dssc_1m():
    from extra_foam.geometries import DSSC_1MGeometryFast
    from extra_foam.geometries import DSSC_1MGeometry

    geom_file = osp.join(_geom_path, "dssc_geo_june19.h5")
    quad_positions = [
        [-124.100,    3.112],
        [-133.068, -110.604],
        [   0.988, -125.236],
        [   4.528,   -4.912]
    ]

    _benchmark_1m_imp(DSSC_1MGeometryFast, DSSC_1MGeometry, geom_file, quad_positions)


def benchmark_lpd_1m():
    from extra_foam.geometries import LPD_1MGeometryFast
    from extra_foam.geometries import LPD_1MGeometry

    geom_file = osp.join(_geom_path, "lpd_mar_18_axesfixed.h5")
    quad_positions = [
        [ 11.4, 299],
        [-11.5,   8],
        [254.5, -16],
        [278.5, 275]
    ]

    _benchmark_1m_imp(LPD_1MGeometryFast, LPD_1MGeometry, geom_file, quad_positions)


def benchmark_agipd_1m():
    from extra_foam.geometries import AGIPD_1MGeometryFast
    from extra_foam.geometries import AGIPD_1MGeometry

    geom_file = osp.join(_geom_path, "agipd_mar18_v11.geom")

    _benchmark_1m_imp(AGIPD_1MGeometryFast, AGIPD_1MGeometry, geom_file)


def benchmark_jungfrau():
    from extra_foam.geometries import JungFrauGeometryFast

    for from_dtype, from_str in _data_sources:
        n_row, n_col = 3, 2
        geom = JungFrauGeometryFast(n_row, n_col)
        n_pulses = 16
        modules = np.ones((n_pulses, n_row * n_col, *geom.module_shape), dtype=from_dtype)

        assembled = geom.output_array_for_position_fast((n_pulses,), _IMAGE_DTYPE)

        t0 = time.perf_counter()
        geom.position_all_modules(modules, assembled)
        dt_assemble = time.perf_counter() - t0

        print(f"\nposition all modules for JungFrauGeometry (from {from_str} data) - \n"
              f"  dt (foam stack only): {dt_assemble:.4f}")

        if modules.dtype == _IMAGE_DTYPE:
            t0 = time.perf_counter()
            geom.dismantle_all_modules(assembled, modules)
            dt_dismantle = time.perf_counter() - t0

            print(f"\ndismantle all modules for JungFrauGeometry (from {from_str} data) - \n"
                  f"  dt (foam stack only): {dt_dismantle:.4f}")

    module = np.ones((n_pulses, *geom.module_shape), dtype=_IMAGE_DTYPE)
    t0 = time.perf_counter()
    JungFrauGeometryFast.maskModule(module)
    dt_mask_cpp = time.perf_counter() - t0

    module = np.ones((n_pulses, *geom.module_shape), dtype=_IMAGE_DTYPE)
    t0 = time.perf_counter()
    JungFrauGeometryFast.mask_module_py(module)
    dt_mask_py = time.perf_counter() - t0

    print(f"\nMask single module for JungFrauGeometry - \n"
          f"  dt (cpp): {dt_mask_cpp:.4f}, dt (py): {dt_mask_py:.4f}")


if __name__ == "__main__":
    print("*" * 80)
    print("Benchmark geometry")
    print("*" * 80)

    benchmark_dssc_1m()

    benchmark_lpd_1m()

    benchmark_agipd_1m()

    benchmark_jungfrau()
