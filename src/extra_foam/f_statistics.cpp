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

#include "xtensor/xnoalias.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "f_statistics.hpp"
#include "f_pyconfig.hpp"

namespace py = pybind11;


// An implementation of Welford's online algorithm for computing variance. This
// implementation also calculates the standard deviation.
template<typename VALUE_TYPE, int N_DIM>
void welford_update(int k,
                    xt::pytensor<VALUE_TYPE, N_DIM>& current_values,
                    xt::pytensor<VALUE_TYPE, N_DIM>& m_values,
                    xt::pytensor<VALUE_TYPE, N_DIM>& s_values,
                    xt::pytensor<VALUE_TYPE, N_DIM>& variance,
                    xt::pytensor<VALUE_TYPE, N_DIM>& std_dev)
{
    xt::pytensor<VALUE_TYPE, N_DIM> old_m_values = m_values;

    // We use xt::noalias() here to ensure that the numpy arrays are modified in-place
    xt::noalias(m_values) += (current_values - m_values) / k;
    xt::noalias(s_values) += (current_values - old_m_values) * (current_values - m_values);
    xt::noalias(variance) = s_values / (k - 1);
    xt::noalias(std_dev) = xt::sqrt(variance);
}

template<typename VALUE_TYPE, int N_DIM>
void update_mean(int k,
                 xt::pytensor<VALUE_TYPE, N_DIM>& current_values,
                 xt::pytensor<VALUE_TYPE, N_DIM>& running_mean)
{
    xt::noalias(running_mean) += (current_values - running_mean) / k;
}

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


#define FOAM_WELFORD_UPDATE_IMP(VALUE_TYPE, N_DIM)           \
  m.def("welford_update", welford_update<VALUE_TYPE, N_DIM>, \
        py::arg("k"),                                        \
        py::arg("current_values").noconvert(),               \
        py::arg("m_values").noconvert(),                     \
        py::arg("s_values").noconvert(),                     \
        py::arg("variance").noconvert(),                     \
        py::arg("std_dev").noconvert());

#define FOAM_WELFORD_UPDATE(VALUE_TYPE)  \
  FOAM_WELFORD_UPDATE_IMP(VALUE_TYPE, 1) \
  FOAM_WELFORD_UPDATE_IMP(VALUE_TYPE, 2) \
  FOAM_WELFORD_UPDATE_IMP(VALUE_TYPE, 3) \
  FOAM_WELFORD_UPDATE_IMP(VALUE_TYPE, 4) \
  FOAM_WELFORD_UPDATE_IMP(VALUE_TYPE, 5)

  FOAM_WELFORD_UPDATE(float)
  FOAM_WELFORD_UPDATE(double)

  m.def("update_mean", update_mean<float, 4>,
        py::arg("k"),
        py::arg("current_values").noconvert(),
        py::arg("running_mean").noconvert());
}
