/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "f_helpers.hpp"

namespace py = pybind11;


PYBIND11_MODULE(helpers, m) {
  m.doc() = "Miscellaneous helper functions in cpp";

  m.def("intersection", &foam::intersection);
}