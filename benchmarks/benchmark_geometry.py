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


_data_sources = [(np.uint16, 'raw'), (np.float32, 'calibrated')]

_geom_path = osp.join(osp.dirname(osp.abspath(__file__)), "../extra_foam/geometries")


def _benchmark_1m_imp(geom_fast_cls, geom_cls, geom_file, quad_positions=None):

    for from_dtype, from_str in _data_sources:
        n_pulses = 64
        modules = np.ones((n_pulses,
                           geom_fast_cls.n_modules,
                           geom_fast_cls.module_shape[0],
                           geom_fast_cls.module_shape[1]), dtype=from_dtype)

        # stack only

        geom = geom_fast_cls()
        out = np.full((n_pulses, *geom.assembledShape()), np.nan, dtype=np.float32)
        t0 = time.perf_counter()
        geom.position_all_modules(modules, out)
        dt_foam_stack = time.perf_counter() - t0

        # assemble with geometry and quad position

        if quad_positions is not None:
            geom = geom_fast_cls.from_h5_file_and_quad_positions(geom_file, quad_positions)
        else:
            geom = geom_fast_cls.from_crystfel_geom(geom_file)
        out = np.full((n_pulses, *geom.assembledShape()), np.nan, dtype=np.float32)
        t0 = time.perf_counter()
        geom.position_all_modules(modules, out)
        dt_foam = time.perf_counter() - t0

        # assemble with geometry and quad position in EXtra-geom

        if quad_positions is not None:
            geom = geom_cls.from_h5_file_and_quad_positions(geom_file, quad_positions)
        else:
            geom = geom_cls.from_crystfel_geom(geom_file)
        out = geom.output_array_for_position_fast((n_pulses,))
        t0 = time.perf_counter()
        geom.position_all_modules(modules, out=out)
        dt_geom = time.perf_counter() - t0

        print(f"\nposition all modules for {geom_cls.__name__} (from {from_str} data) - \n"
              f"  dt (foam stack only): {dt_foam_stack:.4f}, dt (foam): {dt_foam:.4f}, "
              f"dt (geom): {dt_geom:.4f}")


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


if __name__ == "__main__":
    print("*" * 80)
    print("Benchmark geometry")
    print("*" * 80)

    benchmark_dssc_1m()

    benchmark_lpd_1m()

    benchmark_agipd_1m()
