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

PYBIND11_MODULE(azimuthal_integrator, m)
{
  m.doc() = "Azimuthal integration.";

  xt::import_numpy();

  py::enum_<foam::AzimuthalIntegrationMethod>(m, "AzimuthalIntegrationMethod", py::arithmetic())
    .value("Histogram", foam::AzimuthalIntegrationMethod::HISTOGRAM);

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
       &Integrator::integrate1d<const xt::pytensor<DTYPE, 2>&>,                                      \
       py::arg("src").noconvert(), py::arg("npt").noconvert(), py::arg("min_count").noconvert()=1,   \
       py::arg("method")=foam::AzimuthalIntegrationMethod::HISTOGRAM);

  AZIMUTHAL_INTEGRATE1D(double)
  AZIMUTHAL_INTEGRATE1D(float)
}