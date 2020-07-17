/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#if defined(FOAM_USE_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#endif

#include "f_geometry.hpp"
#include "f_pyconfig.hpp"

namespace py = pybind11;


template<typename Detector>
void declareGeometry(py::module &m, const std::string& detector_name)
{
  const std::string py_class_name = detector_name + std::string("Geometry");
  using Geometry = foam::DetectorGeometry<Detector>;

  py::class_<Geometry> cls(m, py_class_name.c_str());

  cls.def(py::init<size_t, size_t, foam::GeometryLayout>(),
          py::arg("n_rows"), py::arg("n_columns"), py::arg("layout") = foam::GeometryLayout::TopRightCW)
    .def(py::init<size_t, size_t, const std::vector<std::array<double, 3>>&, foam::GeometryLayout>(),
         py::arg("n_rows"), py::arg("n_columns"), py::arg("positions"),
         py::arg("layout") = foam::GeometryLayout::TopRightCW)
    .def("nModules", &Geometry::nModules)
    .def("assembledShape", &Geometry::assembledShape)
    .def_readonly_static("pixel_size", &Detector::pixel_size)
    .def_readonly_static("module_shape", &Detector::module_shape)
    .def_readonly_static("asic_shape", &Detector::asic_shape)
    .def_readonly_static("asic_grid_shape", &Detector::asic_grid_shape);

#define FOAM_POSITION_ALL_MODULES(SRC_TYPE, DST_TYPE)                                                            \
  cls.def("positionAllModules",                                                                                  \
    (void (Geometry::*)(const xt::pytensor<SRC_TYPE, 3>&, xt::pytensor<DST_TYPE, 2>&, bool) const)               \
    &Geometry::positionAllModules,                                                                               \
    py::arg("src").noconvert(), py::arg("dst").noconvert(), py::arg("ignore_asic_edge") = false);                \
  cls.def("positionAllModules",                                                                                  \
    (void (Geometry::*)(const std::vector<xt::pytensor<SRC_TYPE, 2>>&, xt::pytensor<DST_TYPE, 2>&, bool) const)  \
    &Geometry::positionAllModules,                                                                               \
    py::arg("src").noconvert(), py::arg("dst").noconvert(), py::arg("ignore_asic_edge") = false);                \
  cls.def("positionAllModules",                                                                                  \
    (void (Geometry::*)(const xt::pytensor<SRC_TYPE, 4>&, xt::pytensor<DST_TYPE, 3>&, bool) const)               \
    &Geometry::positionAllModules,                                                                               \
    py::arg("src").noconvert(), py::arg("dst").noconvert(), py::arg("ignore_asic_edge") = false);                \
  cls.def("positionAllModules",                                                                                  \
    (void (Geometry::*)(const std::vector<xt::pytensor<SRC_TYPE, 3>>&, xt::pytensor<DST_TYPE, 3>&, bool) const)  \
    &Geometry::positionAllModules,                                                                               \
    py::arg("src").noconvert(), py::arg("dst").noconvert(), py::arg("ignore_asic_edge") = false);

  FOAM_POSITION_ALL_MODULES(double, float)
  FOAM_POSITION_ALL_MODULES(float, float)
  FOAM_POSITION_ALL_MODULES(uint16_t, float)
  FOAM_POSITION_ALL_MODULES(int16_t, float) // for ePix100
  FOAM_POSITION_ALL_MODULES(bool, float)
  FOAM_POSITION_ALL_MODULES(uint16_t, uint16_t)
  FOAM_POSITION_ALL_MODULES(bool, bool)

#define FOAM_MASK_MODULE(SRC_TYPE)                                                                 \
  cls.def_static("maskModule",                                                                     \
  static_cast<void (*)(xt::pytensor<SRC_TYPE, 2>&)>(&Geometry::maskModule),                        \
    py::arg("src").noconvert());                                                                   \
  cls.def_static("maskModule",                                                                     \
  static_cast<void (*)(xt::pytensor<SRC_TYPE, 3>&)>(&Geometry::maskModule),                        \
    py::arg("src").noconvert());

  FOAM_MASK_MODULE(float)

#define FOAM_DISMANTLE_ALL_MODULES(SRC_TYPE, DST_TYPE)                                             \
  cls.def("dismantleAllModules",                                                                   \
  (void (Geometry::*)(const xt::pytensor<SRC_TYPE, 2>&, xt::pytensor<DST_TYPE, 3>&) const)         \
    &Geometry::dismantleAllModules,                                                                \
    py::arg("src").noconvert(), py::arg("dst").noconvert());                                       \
  cls.def("dismantleAllModules",                                                                   \
  (void (Geometry::*)(const xt::pytensor<SRC_TYPE, 3>&, xt::pytensor<DST_TYPE, 4>&) const)         \
    &Geometry::dismantleAllModules,                                                                \
    py::arg("src").noconvert(), py::arg("dst").noconvert());

  FOAM_DISMANTLE_ALL_MODULES(float, float)
  FOAM_DISMANTLE_ALL_MODULES(uint16_t, uint16_t)
  FOAM_DISMANTLE_ALL_MODULES(bool, bool)
}

PYBIND11_MODULE(geometry, m)
{
  xt::import_numpy();

  m.doc() = "Generalized detector geometry.";

  py::enum_<foam::GeometryLayout>(m, "GeometryLayout", py::arithmetic())
    .value("TopRightCW", foam::GeometryLayout::TopRightCW)
    .value("BottomRightCCW", foam::GeometryLayout::BottomRightCCW)
    .value("BottomLeftCW", foam::GeometryLayout::BottomLeftCW)
    .value("TopLeftCCW", foam::GeometryLayout::TopLeftCCW);

  declareGeometry<foam::JungFrau>(m, "JungFrau");
  declareGeometry<foam::EPix100>(m, "EPix100");
}
