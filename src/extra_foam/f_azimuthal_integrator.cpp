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

#include "f_azimuthal_integrator.hpp"
#include "f_pyconfig.hpp"

namespace py = pybind11;

void declareAzimuthalIntegrator(py::module& m)
{
  using Integrator = foam::AzimuthalIntegrator;

  std::string py_class_name = "AzimuthalIntegrator";
  py::class_<Integrator> cls(m, py_class_name.c_str());

  cls.def(py::init<double, double, double, double, double, double>(),
          py::arg("dist"), py::arg("poni1"), py::arg("poni2"),
          py::arg("pixel1"), py::arg("pixel2"), py::arg("wavelength"));

#define AZIMUTHAL_INTEGRATE1D(DTYPE)                                                                 \
  cls.def("integrate1d", (std::pair<foam::ReducedVectorType<xt::pytensor<DTYPE, 2>>,                 \
                                    foam::ReducedVectorType<xt::pytensor<DTYPE, 2>>>                 \
                          (Integrator::*)(const xt::pytensor<DTYPE, 2>&, size_t, size_t,             \
                                          foam::AzimuthalIntegrationMethod) const)                   \
     &Integrator::integrate1d<const xt::pytensor<DTYPE, 2>&>,                                        \
     py::arg("src").noconvert(), py::arg("npt").noconvert(), py::arg("min_count").noconvert()=1,     \
     py::arg("method")=foam::AzimuthalIntegrationMethod::HISTOGRAM);

  AZIMUTHAL_INTEGRATE1D(double)
  AZIMUTHAL_INTEGRATE1D(float)
}

void declareConcentricRingFinder(py::module& m)
{
  using Finder = foam::ConcentricRingFinder;

  std::string py_class_name = "ConcentricRingFinder";
  py::class_<Finder> cls(m, py_class_name.c_str());

  cls.def(py::init<double, double>(), py::arg("pixel_x"), py::arg("pixel_y"));

#define CONCENTRIC_RING_FINDER_SEARCH(DTYPE)                                                            \
  cls.def("search", (std::array<double, 2>                                                              \
                     (Finder::*)(const xt::pytensor<DTYPE, 2>&, double, double, size_t, size_t) const)  \
     &Finder::search<const xt::pytensor<DTYPE, 2>&>,                                                    \
     py::arg("src").noconvert(), py::arg("cx0"), py::arg("cy0"),                                        \
     py::arg("n_grids").noconvert() = 128, py::arg("min_cunt").noconvert() = 1);

  CONCENTRIC_RING_FINDER_SEARCH(double)
  CONCENTRIC_RING_FINDER_SEARCH(float)
}


PYBIND11_MODULE(azimuthal_integrator, m)
{
  m.doc() = "Azimuthal integration.";

  xt::import_numpy();

  py::enum_<foam::AzimuthalIntegrationMethod>(m, "AzimuthalIntegrationMethod", py::arithmetic())
    .value("Histogram", foam::AzimuthalIntegrationMethod::HISTOGRAM);

  declareAzimuthalIntegrator(m);

  declareConcentricRingFinder(m);

}