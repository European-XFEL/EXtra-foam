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

#include "xtensor/xmath.hpp"

#include "f_imageproc.hpp"
#include "f_pyconfig.hpp"

namespace py = pybind11;


PYBIND11_MODULE(imageproc, m)
{
  xt::import_numpy();

  using namespace foam;

  m.doc() = "A collection of image processing functions.";

#define FOAM_NANMEAN_IMAGE_ARRAY_IMPL(VALUE_TYPE)                                      \
  m.def("nanmeanImageArray", [] (const xt::pytensor<VALUE_TYPE, 3>& src)               \
    { return nanmeanImageArray(src); }, py::arg("src").noconvert());

#define FOAM_NANMEAN_IMAGE_ARRAY_WITH_FILTER_IMPL(VALUE_TYPE)                                   \
  m.def("nanmeanImageArray",                                                                    \
    [] (const xt::pytensor<VALUE_TYPE, 3>& src, const std::vector<size_t>& keep)                \
    { return nanmeanImageArray(src, keep); }, py::arg("src").noconvert(), py::arg("keep"));

#define FOAM_NANMEAN_IMAGE_ARRAY_BINARY_IMPL(VALUE_TYPE)                                        \
  m.def("nanmeanImageArray",                                                                    \
    [] (const xt::pytensor<VALUE_TYPE, 2>& src1, const xt::pytensor<VALUE_TYPE, 2>& src2)       \
    { return nanmeanImageArray(src1, src2); },                                                  \
    py::arg("src1").noconvert(), py::arg("src2").noconvert());

  FOAM_NANMEAN_IMAGE_ARRAY_IMPL(double)
  FOAM_NANMEAN_IMAGE_ARRAY_IMPL(float)
  FOAM_NANMEAN_IMAGE_ARRAY_WITH_FILTER_IMPL(double)
  FOAM_NANMEAN_IMAGE_ARRAY_WITH_FILTER_IMPL(float)
  FOAM_NANMEAN_IMAGE_ARRAY_BINARY_IMPL(double)
  FOAM_NANMEAN_IMAGE_ARRAY_BINARY_IMPL(float)

#define FOAM_MOVING_AVG_IMAGE_DATA_IMPL(VALUE_TYPE, N_DIM)                                     \
  m.def("movingAvgImageData",                                                                  \
    &movingAvgImageData<xt::pytensor<VALUE_TYPE, N_DIM>>,                                      \
    py::arg("src").noconvert(), py::arg("data").noconvert(), py::arg("count"));

  FOAM_MOVING_AVG_IMAGE_DATA_IMPL(double, 2)
  FOAM_MOVING_AVG_IMAGE_DATA_IMPL(float, 2)
  FOAM_MOVING_AVG_IMAGE_DATA_IMPL(double, 3)
  FOAM_MOVING_AVG_IMAGE_DATA_IMPL(float, 3)

#define FOAM_MASK_IMAGE_DATA_IMPL(FUNCTOR, VALUE_TYPE, N_DIM)                                 \
  m.def(#FUNCTOR,                                                                             \
    &FUNCTOR<xt::pytensor<VALUE_TYPE, N_DIM>>, py::arg("src").noconvert());

#define FOAM_MASK_IMAGE_DATA(FUNCTOR)                                                         \
  FOAM_MASK_IMAGE_DATA_IMPL(FUNCTOR, double, 2)                                               \
  FOAM_MASK_IMAGE_DATA_IMPL(FUNCTOR, float, 2)                                                \
  FOAM_MASK_IMAGE_DATA_IMPL(FUNCTOR, double, 3)                                               \
  FOAM_MASK_IMAGE_DATA_IMPL(FUNCTOR, float, 3)

  FOAM_MASK_IMAGE_DATA(maskZeroImageData)
  FOAM_MASK_IMAGE_DATA(maskNanImageData)

#define FOAM_MASK_IMAGE_DATA_THRESHOLD_IMPL(FUNCTOR, VALUE_TYPE, N_DIM)                      \
  m.def(#FUNCTOR,                                                                         \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&, VALUE_TYPE, VALUE_TYPE))                     \
    &FUNCTOR<xt::pytensor<VALUE_TYPE, N_DIM>, VALUE_TYPE>,                                   \
    py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

#define FOAM_MASK_IMAGE_DATA_THRESHOLD(FUNCTOR)                                              \
  FOAM_MASK_IMAGE_DATA_THRESHOLD_IMPL(FUNCTOR, double, 2)                                    \
  FOAM_MASK_IMAGE_DATA_THRESHOLD_IMPL(FUNCTOR, float, 2)                                     \
  FOAM_MASK_IMAGE_DATA_THRESHOLD_IMPL(FUNCTOR, double, 3)                                    \
  FOAM_MASK_IMAGE_DATA_THRESHOLD_IMPL(FUNCTOR, float, 3)

  FOAM_MASK_IMAGE_DATA_THRESHOLD(maskZeroImageData)
  FOAM_MASK_IMAGE_DATA_THRESHOLD(maskNanImageData)

#define FOAM_MASK_IMAGE_DATA_IMAGE_IMPL(FUNCTOR, VALUE_TYPE, N_DIM)                        \
  m.def(#FUNCTOR,                                                                         \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&, const xt::pytensor<bool, 2>&))             \
    &FUNCTOR<xt::pytensor<VALUE_TYPE, N_DIM>, xt::pytensor<bool, 2>>,                      \
    py::arg("src").noconvert(), py::arg("mask").noconvert());

#define FOAM_MASK_IMAGE_DATA_IMAGE(FUNCTOR)                                                \
  FOAM_MASK_IMAGE_DATA_IMAGE_IMPL(FUNCTOR, double, 2)                                      \
  FOAM_MASK_IMAGE_DATA_IMAGE_IMPL(FUNCTOR, float, 2)                                       \
  FOAM_MASK_IMAGE_DATA_IMAGE_IMPL(FUNCTOR, double, 3)                                      \
  FOAM_MASK_IMAGE_DATA_IMAGE_IMPL(FUNCTOR, float, 3)

  FOAM_MASK_IMAGE_DATA_IMAGE(maskZeroImageData)
  FOAM_MASK_IMAGE_DATA_IMAGE(maskNanImageData)

#define FOAM_MASK_IMAGE_DATA_BOTH_IMPL(FUNCTOR, VALUE_TYPE, N_DIM)                                      \
  m.def(#FUNCTOR,                                                                                       \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&, const xt::pytensor<bool, 2>&, VALUE_TYPE, VALUE_TYPE))  \
    &FUNCTOR<xt::pytensor<VALUE_TYPE, N_DIM>, xt::pytensor<bool, 2>, VALUE_TYPE>,                       \
    py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));

#define FOAM_MASK_IMAGE_DATA_BOTH(FUNCTOR)                                                \
  FOAM_MASK_IMAGE_DATA_BOTH_IMPL(FUNCTOR, double, 2)                                      \
  FOAM_MASK_IMAGE_DATA_BOTH_IMPL(FUNCTOR, float, 2)                                       \
  FOAM_MASK_IMAGE_DATA_BOTH_IMPL(FUNCTOR, double, 3)                                      \
  FOAM_MASK_IMAGE_DATA_BOTH_IMPL(FUNCTOR, float, 3)

  FOAM_MASK_IMAGE_DATA_BOTH(maskZeroImageData)
  FOAM_MASK_IMAGE_DATA_BOTH(maskNanImageData)

#define FOAM_MASK_IMAGE_DATA_AND_MASK_IMPL(VALUE_TYPE)                                       \
  m.def("maskImageData",                                                                     \
    (void (*)(xt::pytensor<VALUE_TYPE, 2>&, xt::pytensor<bool, 2>&))                         \
    &maskImageData<xt::pytensor<VALUE_TYPE, 2>, xt::pytensor<bool, 2>>,                      \
    py::arg("src").noconvert(), py::arg("mask").noconvert());

  FOAM_MASK_IMAGE_DATA_AND_MASK_IMPL(double)
  FOAM_MASK_IMAGE_DATA_AND_MASK_IMPL(float)

#define FOAM_MASK_IMAGE_DATA_BOTH_AND_MASK_IMPL(VALUE_TYPE)                                  \
  m.def("maskImageData",                                                                     \
    (void (*)(xt::pytensor<VALUE_TYPE, 2>&, xt::pytensor<bool, 2>&, VALUE_TYPE, VALUE_TYPE)) \
    &maskImageData<xt::pytensor<VALUE_TYPE, 2>, xt::pytensor<bool, 2>, VALUE_TYPE>,          \
    py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));

  FOAM_MASK_IMAGE_DATA_BOTH_AND_MASK_IMPL(double)
  FOAM_MASK_IMAGE_DATA_BOTH_AND_MASK_IMPL(float)

#define FOAM_CORRECT_OFFSET_IMPL(VALUE_TYPE, N_DIM)                                         \
  m.def("correctOffset",                                                                    \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&, const xt::pytensor<VALUE_TYPE, N_DIM>&))    \
    &correctImageData<OffsetPolicy, xt::pytensor<VALUE_TYPE, N_DIM>>,                       \
    py::arg("src").noconvert(), py::arg("offset").noconvert());

  FOAM_CORRECT_OFFSET_IMPL(double, 2)
  FOAM_CORRECT_OFFSET_IMPL(float, 2)
  FOAM_CORRECT_OFFSET_IMPL(double, 3)
  FOAM_CORRECT_OFFSET_IMPL(float, 3)

#define FOAM_CORRECT_GAIN_IMPL(VALUE_TYPE, N_DIM)                                           \
  m.def("correctGain",                                                                      \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&, const xt::pytensor<VALUE_TYPE, N_DIM>&))    \
    &correctImageData<GainPolicy, xt::pytensor<VALUE_TYPE, N_DIM>>,                         \
    py::arg("src").noconvert(), py::arg("gain").noconvert());

  FOAM_CORRECT_GAIN_IMPL(double, 2)
  FOAM_CORRECT_GAIN_IMPL(float, 2)
  FOAM_CORRECT_GAIN_IMPL(double, 3)
  FOAM_CORRECT_GAIN_IMPL(float, 3)

#define FOAM_CORRECT_GAIN_AND_OFFSET_IMPL(VALUE_TYPE, N_DIM)                                    \
  m.def("correctGainOffset",                                                                    \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&,                                                 \
              const xt::pytensor<VALUE_TYPE, N_DIM>&, const xt::pytensor<VALUE_TYPE, N_DIM>&))  \
    &correctImageData<xt::pytensor<VALUE_TYPE, N_DIM>>,                                         \
    py::arg("src").noconvert(), py::arg("gain").noconvert(), py::arg("offset").noconvert());

  FOAM_CORRECT_GAIN_AND_OFFSET_IMPL(double, 2)
  FOAM_CORRECT_GAIN_AND_OFFSET_IMPL(float, 2)
  FOAM_CORRECT_GAIN_AND_OFFSET_IMPL(double, 3)
  FOAM_CORRECT_GAIN_AND_OFFSET_IMPL(float, 3)
}
