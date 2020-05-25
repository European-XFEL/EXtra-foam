import os.path as osp

import pytest

import numpy as np

from extra_data.stacking import StackView

from extra_foam.geometries import EPix100GeometryFast, JungFrauGeometryFast, load_geometry
from extra_foam.config import config


_geom_path = osp.join(osp.dirname(osp.abspath(__file__)), "../")

_IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']
_RAW_IMAGE_DTYPE = config['SOURCE_RAW_IMAGE_DTYPE']


class TestJungFrauGeometryFast:
    """Test pulse-resolved."""
    @classmethod
    def setup_class(cls):
        cls.n_pulses = 2
        cls.module_shape = (512, 1024)
        cls.asic_shape = (256, 256)

        cls.geom_21_stack = load_geometry("JungFrau", stack_only=True, n_modules=2)
        cls.geom_32_stack = load_geometry("JungFrau", stack_only=True, n_modules=6)

        cls.cases = [
            (cls.geom_21_stack, 2, (1024, 1024)),
            (cls.geom_32_stack, 6, (1536, 2048)),
        ]

        # TODO: add default JungFrau geometries
        geom_file = osp.join(osp.expanduser("~"), "jungfrau.geom")
        try:
            cls.geom_32_cfel = load_geometry("JungFrau", filepath=geom_file, n_modules=6)
        except FileNotFoundError:
            module_coordinates = [
                np.array([ 0.08452896,  0.07981445, 0.]),
                np.array([ 0.08409096,  0.03890507, 0.]),
                np.array([ 0.08385471, -0.00210121, 0.]),
                np.array([-0.08499321, -0.04048030, 0.]),
                np.array([-0.08477046,  0.00059965, 0.]),
                np.array([-0.08479671,  0.04162323, 0.])
            ]

            cls.geom_32_cfel = JungFrauGeometryFast(3, 2, module_coordinates)
        cls.cases.append((cls.geom_32_cfel, 6, (1607, 2260)))

    @pytest.mark.parametrize("dtype", [_IMAGE_DTYPE, _RAW_IMAGE_DTYPE, bool])
    def testAssemblingArray(self, dtype):
        for geom, n_modules, assembled_shape_gt in self.cases:
            modules = np.ones((self.n_pulses, n_modules, *self.module_shape), dtype=dtype)

            assembled = geom.output_array_for_position_fast((self.n_pulses,), _IMAGE_DTYPE)
            geom.position_all_modules(modules, assembled)
            assert assembled_shape_gt == assembled.shape[-2:]

            # test dismantle
            dismantled = geom.output_array_for_dismantle_fast((self.n_pulses,), _IMAGE_DTYPE)
            geom.dismantle_all_modules(assembled, dismantled)
            np.testing.assert_array_equal(modules, dismantled)

    @pytest.mark.parametrize("dtype", [_IMAGE_DTYPE, _RAW_IMAGE_DTYPE, bool])
    def testAssemblingVector(self, dtype):
        for geom, n_modules, assembled_shape_gt in self.cases:
            modules = StackView(
                {i: np.ones((self.n_pulses, *self.module_shape), dtype=dtype) for i in range(n_modules)},
                n_modules,
                (self.n_pulses, ) + self.module_shape,
                dtype,
                np.nan)

            assembled = geom.output_array_for_position_fast((self.n_pulses,), _IMAGE_DTYPE)
            geom.position_all_modules(modules, assembled)
            assert assembled_shape_gt == assembled.shape[-2:]

    @pytest.mark.parametrize("dtype", [_IMAGE_DTYPE, _RAW_IMAGE_DTYPE])
    def testAssemblingArrayWithAsicEdgeIgnored(self, dtype):
        ah, aw = self.asic_shape[0], self.asic_shape[1]

        # assembling with a geometry file is not tested
        for geom, n_modules, assembled_shape_gt in self.cases[:-1]:
            modules = np.ones((self.n_pulses, n_modules, *self.module_shape), dtype=dtype)

            assembled = geom.output_array_for_position_fast((self.n_pulses,), _IMAGE_DTYPE)
            geom.position_all_modules(modules, assembled, ignore_asic_edge=True)

            assert 0 == np.count_nonzero(~np.isnan(assembled[:, :, 0::aw]))
            assert 0 == np.count_nonzero(~np.isnan(assembled[:, :, aw - 1::aw]))
            assert 0 == np.count_nonzero(~np.isnan(assembled[:, 0::ah, :]))
            assert 0 == np.count_nonzero(~np.isnan(assembled[:, ah - 1::ah, :]))

    def testMaskModule(self):
        module1 = np.ones((self.n_pulses, *self.module_shape), dtype=_IMAGE_DTYPE)
        JungFrauGeometryFast.maskModule(module1)
        module2 = np.copy(module1)
        JungFrauGeometryFast.mask_module_py(module2)

        np.testing.assert_array_equal(module1, module2)


class TestEpix100GeometryFast:
    """Test train-resolved."""
    @classmethod
    def setup_class(cls):
        cls.module_shape = (708, 768)
        cls.asic_shape = (354, 384)

        cls.geom_21_stack = load_geometry("ePix100", stack_only=True, n_modules=2)
        cls.geom_22_stack = load_geometry("ePix100", stack_only=True, n_modules=4)

        cls.cases = [
            (cls.geom_21_stack, 2, (1416, 768)),
            (cls.geom_22_stack, 4, (1416, 1536)),
        ]

    @pytest.mark.parametrize("dtype", [_IMAGE_DTYPE, _RAW_IMAGE_DTYPE, np.int16, bool])
    def testAssemblingArray(self, dtype):
        for geom, n_modules, assembled_shape_gt in self.cases:
            modules = np.ones((n_modules, *self.module_shape), dtype=dtype)

            assembled = geom.output_array_for_position_fast(dtype=_IMAGE_DTYPE)
            geom.position_all_modules(modules, assembled)
            assert assembled_shape_gt == assembled.shape[-2:]

            # test dismantle
            dismantled = geom.output_array_for_dismantle_fast(dtype=_IMAGE_DTYPE)
            geom.dismantle_all_modules(assembled, dismantled)
            np.testing.assert_array_equal(modules, dismantled)

    @pytest.mark.parametrize("dtype", [_IMAGE_DTYPE, _RAW_IMAGE_DTYPE, np.int16, bool])
    def testAssemblingVector(self, dtype):
        for geom, n_modules, assembled_shape_gt in self.cases:
            modules = StackView(
                {i: np.ones(self.module_shape, dtype=dtype) for i in range(n_modules)},
                n_modules,
                self.module_shape,
                dtype,
                np.nan)

            assembled = geom.output_array_for_position_fast(dtype=_IMAGE_DTYPE)
            geom.position_all_modules(modules, assembled)
            assert assembled_shape_gt == assembled.shape[-2:]

    @pytest.mark.parametrize("dtype", [_IMAGE_DTYPE, _RAW_IMAGE_DTYPE, np.int16])
    def testAssemblingWithAsicEdgeIgnored(self, dtype):
        ah, aw = self.asic_shape[0], self.asic_shape[1]

        for geom, n_modules, assembled_shape_gt in self.cases:
            modules = np.ones((n_modules, *self.module_shape), dtype=dtype)

            assembled = geom.output_array_for_position_fast(dtype=_IMAGE_DTYPE)
            geom.position_all_modules(modules, assembled, ignore_asic_edge=True)

            assert 0 == np.count_nonzero(~np.isnan(assembled[:, 0::aw]))
            assert 0 == np.count_nonzero(~np.isnan(assembled[:, aw - 1::aw]))
            assert 0 == np.count_nonzero(~np.isnan(assembled[0::ah, :]))
            assert 0 == np.count_nonzero(~np.isnan(assembled[ah - 1::ah, :]))

    def testMaskModule(self):
        module1 = np.ones(self.module_shape, dtype=_IMAGE_DTYPE)
        EPix100GeometryFast.maskModule(module1)
        module2 = np.copy(module1)
        EPix100GeometryFast.mask_module_py(module2)

        np.testing.assert_array_equal(module1, module2)
