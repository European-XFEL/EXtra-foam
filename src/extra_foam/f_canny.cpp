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

#include "f_canny.hpp"
#include "f_pyconfig.hpp"

namespace py = pybind11;


PYBIND11_MODULE(canny, m)
{
  m.doc() = "Canny edge detection implementation.";

  xt::import_numpy();

#define FOAM_CANNY_EDGE(INPUT_TYPE, RETURN_TYPE)                                                                              \
  m.def("cannyEdge",                                                                                             \
    (void (*)(const xt::pytensor<INPUT_TYPE, 2>& src, xt::pytensor<RETURN_TYPE, 2>& dst, double lb, double ub))  \
    &foam::cannyEdge<xt::pytensor<INPUT_TYPE, 2>, xt::pytensor<RETURN_TYPE, 2>>,                                 \
    py::arg("src").noconvert(), py::arg("dst").noconvert(),                                                      \
    py::arg("lb") = std::numeric_limits<double>::min(), py::arg("ub") = std::numeric_limits<double>::max());

  FOAM_CANNY_EDGE(double, double);
  FOAM_CANNY_EDGE(float, float);
  FOAM_CANNY_EDGE(double, uint8_t);
  FOAM_CANNY_EDGE(float, uint8_t);
}
