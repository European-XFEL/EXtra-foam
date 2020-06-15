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

#include "f_smooth.hpp"
#include "f_pyconfig.hpp"

namespace py = pybind11;


PYBIND11_MODULE(smooth, m)
{
  m.doc() = "Data smooth implementation.";

  xt::import_numpy();

#define FOAM_GAUSSIAN_BLUR(INPUT_TYPE)                                                                           \
  m.def("gaussianBlur",                                                                                          \
    (void (*)(const xt::pytensor<INPUT_TYPE, 2>& src, xt::pytensor<INPUT_TYPE, 2>& dst, size_t, double))         \
    &foam::gaussianBlur<xt::pytensor<INPUT_TYPE, 2>>,                               \
    py::arg("src").noconvert(), py::arg("dst").noconvert(), py::arg("k_size"), py::arg("sigma") = -1);

  FOAM_GAUSSIAN_BLUR(double);
  FOAM_GAUSSIAN_BLUR(float);
  FOAM_GAUSSIAN_BLUR(uint16_t);
  FOAM_GAUSSIAN_BLUR(int16_t);

}