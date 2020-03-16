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

#if defined(FOAM_WITH_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#endif

#include "f_geometry.hpp"
#include "f_pyconfig.hpp"

namespace py = pybind11;


template<typename Geometry>
void declare_1MGeometry(py::module &m, std::string&& detector)
{
  using GeometryBase = foam::Detector1MGeometryBase<Geometry>;
  const std::string py_base_class_name = detector + std::string("_Detector1MGeometryBase");

  py::class_<GeometryBase> base(m, py_base_class_name.c_str());

  base.def("positionAllModules",
    (void (GeometryBase::*)(const xt::pytensor<float, 4>&, xt::pytensor<float, 3>&) const)
    &GeometryBase::positionAllModules,
    py::arg("src").noconvert(), py::arg("dst").noconvert());
  base.def("positionAllModules",
    (void (GeometryBase::*)(const xt::pytensor<uint16_t, 4>&, xt::pytensor<float, 3>&) const)
    &GeometryBase::positionAllModules,
    py::arg("src").noconvert(), py::arg("dst").noconvert());
  base.def("positionAllModules",
    (void (GeometryBase::*)(const std::vector<xt::pytensor<float, 3>>&, xt::pytensor<float, 3>&) const)
    &GeometryBase::positionAllModules,
    py::arg("src").noconvert(), py::arg("dst").noconvert());
  base.def("positionAllModules",
    (void (GeometryBase::*)(const std::vector<xt::pytensor<uint16_t, 3>>&, xt::pytensor<float, 3>&) const)
    &GeometryBase::positionAllModules,
    py::arg("src").noconvert(), py::arg("dst").noconvert());
  base.def("assembledShape", &GeometryBase::assembledShape)
    .def_readonly_static("n_quads", &GeometryBase::n_quads)
    .def_readonly_static("n_modules", &GeometryBase::n_modules)
    .def_readonly_static("n_modules_per_quad", &GeometryBase::n_modules_per_quad);

  const std::string py_class_name = detector + std::string("_1MGeometry");

  py::class_<Geometry, GeometryBase> cls(m, py_class_name.c_str());

  cls.def(py::init())
    .def(py::init<const std::array<std::array<std::array<double, 3>, Geometry::n_tiles_per_module>, Geometry::n_modules> &>())
    .def_static("pixelSize", []() { return xt::pytensor<double, 1>(Geometry::pixelSize()); } )
    .def_readonly_static("module_shape", &Geometry::module_shape)
    .def_readonly_static("tile_shape", &Geometry::tile_shape)
    .def_readonly_static("n_tiles_per_module", &Geometry::n_tiles_per_module)
    .def_readonly_static("quad_orientations", &Geometry::quad_orientations);

}

PYBIND11_MODULE(geometry, m)
{
  xt::import_numpy();

  m.doc() = "Detector geometry.";

  declare_1MGeometry<foam::AGIPD_1MGeometry>(m, "AGIPD");

  declare_1MGeometry<foam::LPD_1MGeometry>(m, "LPD");

  declare_1MGeometry<foam::DSSC_1MGeometry>(m, "DSSC");
}
