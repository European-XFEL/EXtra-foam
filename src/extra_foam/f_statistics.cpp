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

#include "f_statistics.hpp"
#include "f_pyconfig.hpp"

namespace py = pybind11;


PYBIND11_MODULE(statistics, m)
{
  xt::import_numpy();

  m.doc() = "A collection of statistics functions.";

  m.def("nansum", [] (const xt::pytensor<double, 2>& src) { return foam::nansum(src); });
  m.def("nansum", [] (const xt::pytensor<float, 2>& src) { return foam::nansum(src); });

  m.def("nanmean", [] (const xt::pytensor<double, 2>& src) { return foam::nanmean(src); });
  m.def("nanmean", [] (const xt::pytensor<float, 2>& src) { return foam::nanmean(src); });
}
