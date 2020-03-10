/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "f_statistics.hpp"
#include "f_pyconfig.hpp"

namespace py = pybind11;


PYBIND11_MODULE(statistics, m)
{

  using namespace foam;

  xt::import_numpy();

  m.doc() = "A collection of statistics functions.";

#define FOAM_NAN_REDUCER_IMP(REDUCER, VALUE_TYPE, N_DIM)                                          \
  m.def(#REDUCER, [] (const xt::pytensor<VALUE_TYPE, N_DIM>& src, const std::vector<int>& axis)   \
  {                                                                                               \
    return xt::eval(xt::REDUCER<VALUE_TYPE>(src, axis));                                          \
  }, py::arg("src").noconvert(), py::arg("axis"));                                                \
  m.def(#REDUCER, [] (const xt::pytensor<VALUE_TYPE, N_DIM>& src, int axis)                       \
  {                                                                                               \
    return xt::eval(xt::REDUCER<VALUE_TYPE>(src, {axis}));                                        \
  }, py::arg("src").noconvert(), py::arg("axis"));                                                \
  m.def(#REDUCER, [] (const xt::pytensor<VALUE_TYPE, N_DIM>& src)                                 \
  {                                                                                               \
    return xt::eval(xt::REDUCER<VALUE_TYPE>(src))[0];                                             \
  }, py::arg("src").noconvert());

#define FOAM_NAN_REDUCER_ALL_DIMENSIONS(FUNCTOR, VALUE_TYPE)                                   \
  FOAM_NAN_REDUCER_IMP(FUNCTOR, VALUE_TYPE, 1)                                                 \
  FOAM_NAN_REDUCER_IMP(FUNCTOR, VALUE_TYPE, 2)                                                 \
  FOAM_NAN_REDUCER_IMP(FUNCTOR, VALUE_TYPE, 3)                                                 \
  FOAM_NAN_REDUCER_IMP(FUNCTOR, VALUE_TYPE, 4)                                                 \
  FOAM_NAN_REDUCER_IMP(FUNCTOR, VALUE_TYPE, 5)

#define FOAM_NAN_REDUCER(FUNCTOR)                                                              \
  FOAM_NAN_REDUCER_ALL_DIMENSIONS(FUNCTOR, float)                                              \
  FOAM_NAN_REDUCER_ALL_DIMENSIONS(FUNCTOR, double)

  FOAM_NAN_REDUCER(nansum)
  FOAM_NAN_REDUCER(nanmean)

}
