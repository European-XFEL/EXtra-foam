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

  //
  // mask
  //

#define FOAM_IMAGE_DATA_NAN_MASK_IMPL(VALUE_TYPE, N_DIM)                                      \
  m.def("imageDataNanMask",                                                                   \
    &imageDataNanMask<xt::pytensor<VALUE_TYPE, N_DIM>, xt::pytensor<bool, N_DIM>>,            \
    py::arg("src").noconvert(), py::arg("out").noconvert());

  FOAM_IMAGE_DATA_NAN_MASK_IMPL(double, 2)
  FOAM_IMAGE_DATA_NAN_MASK_IMPL(float, 2)

#define FOAM_MASK_IMAGE_DATA_IMPL(FUNCTOR, VALUE_TYPE, N_DIM)                                 \
  m.def(#FUNCTOR,                                                                             \
    &FUNCTOR<xt::pytensor<VALUE_TYPE, N_DIM>>, py::arg("src").noconvert());

#define FOAM_MASK_IMAGE_DATA(FUNCTOR)                                                         \
  FOAM_MASK_IMAGE_DATA_IMPL(FUNCTOR, double, 2)                                               \
  FOAM_MASK_IMAGE_DATA_IMPL(FUNCTOR, float, 2)                                                \
  FOAM_MASK_IMAGE_DATA_IMPL(FUNCTOR, double, 3)                                               \
  FOAM_MASK_IMAGE_DATA_IMPL(FUNCTOR, float, 3)

  FOAM_MASK_IMAGE_DATA(maskImageDataZero)
  FOAM_MASK_IMAGE_DATA(maskImageDataNan)

#define FOAM_MASK_IMAGE_DATA_THRESHOLD_IMPL(FUNCTOR, VALUE_TYPE, N_DIM)                      \
  m.def(#FUNCTOR,                                                                            \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&, VALUE_TYPE, VALUE_TYPE))                     \
    &FUNCTOR<xt::pytensor<VALUE_TYPE, N_DIM>, VALUE_TYPE>,                                   \
    py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

#define FOAM_MASK_IMAGE_DATA_THRESHOLD(FUNCTOR)                                              \
  FOAM_MASK_IMAGE_DATA_THRESHOLD_IMPL(FUNCTOR, double, 2)                                    \
  FOAM_MASK_IMAGE_DATA_THRESHOLD_IMPL(FUNCTOR, float, 2)                                     \
  FOAM_MASK_IMAGE_DATA_THRESHOLD_IMPL(FUNCTOR, double, 3)                                    \
  FOAM_MASK_IMAGE_DATA_THRESHOLD_IMPL(FUNCTOR, float, 3)

  FOAM_MASK_IMAGE_DATA_THRESHOLD(maskImageDataZero)
  FOAM_MASK_IMAGE_DATA_THRESHOLD(maskImageDataNan)

#define FOAM_MASK_IMAGE_DATA_THRESHOLD_WITH_OUT_IMPL(FUNCTOR, VALUE_TYPE, N_DIM)                      \
  m.def(#FUNCTOR,                                                                                     \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&, VALUE_TYPE, VALUE_TYPE, xt::pytensor<bool, N_DIM>&))  \
    &FUNCTOR<xt::pytensor<VALUE_TYPE, N_DIM>, VALUE_TYPE, xt::pytensor<bool, N_DIM>>,                 \
    py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"), py::arg("out").noconvert());

#define FOAM_MASK_IMAGE_DATA_THRESHOLD_WITH_OUT(FUNCTOR)                                              \
  FOAM_MASK_IMAGE_DATA_THRESHOLD_WITH_OUT_IMPL(FUNCTOR, double, 2)                                    \
  FOAM_MASK_IMAGE_DATA_THRESHOLD_WITH_OUT_IMPL(FUNCTOR, float, 2)

  FOAM_MASK_IMAGE_DATA_THRESHOLD_WITH_OUT(maskImageDataZero)
  FOAM_MASK_IMAGE_DATA_THRESHOLD_WITH_OUT(maskImageDataNan)

#define FOAM_MASK_IMAGE_DATA_IMAGE_IMPL(FUNCTOR, VALUE_TYPE, N_DIM)                        \
  m.def(#FUNCTOR,                                                                          \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&, const xt::pytensor<bool, 2>&))             \
    &FUNCTOR<xt::pytensor<VALUE_TYPE, N_DIM>, xt::pytensor<bool, 2>>,                      \
    py::arg("src").noconvert(), py::arg("mask").noconvert());

#define FOAM_MASK_IMAGE_DATA_IMAGE(FUNCTOR)                                                \
  FOAM_MASK_IMAGE_DATA_IMAGE_IMPL(FUNCTOR, double, 2)                                      \
  FOAM_MASK_IMAGE_DATA_IMAGE_IMPL(FUNCTOR, float, 2)                                       \
  FOAM_MASK_IMAGE_DATA_IMAGE_IMPL(FUNCTOR, double, 3)                                      \
  FOAM_MASK_IMAGE_DATA_IMAGE_IMPL(FUNCTOR, float, 3)

  FOAM_MASK_IMAGE_DATA_IMAGE(maskImageDataZero)
  FOAM_MASK_IMAGE_DATA_IMAGE(maskImageDataNan)

#define FOAM_MASK_IMAGE_DATA_IMAGE_WITH_OUT_IMPL(FUNCTOR, VALUE_TYPE, N_DIM)                                \
  m.def(#FUNCTOR,                                                                                           \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&, const xt::pytensor<bool, 2>&, xt::pytensor<bool, N_DIM>&))  \
    &FUNCTOR<xt::pytensor<VALUE_TYPE, N_DIM>, xt::pytensor<bool, 2>, xt::pytensor<bool, N_DIM>>,            \
    py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("out").noconvert());

#define FOAM_MASK_IMAGE_DATA_IMAGE_WITH_OUT(FUNCTOR)                                                \
  FOAM_MASK_IMAGE_DATA_IMAGE_WITH_OUT_IMPL(FUNCTOR, double, 2)                                      \
  FOAM_MASK_IMAGE_DATA_IMAGE_WITH_OUT_IMPL(FUNCTOR, float, 2)

  FOAM_MASK_IMAGE_DATA_IMAGE_WITH_OUT(maskImageDataZero)
  FOAM_MASK_IMAGE_DATA_IMAGE_WITH_OUT(maskImageDataNan)

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

  FOAM_MASK_IMAGE_DATA_BOTH(maskImageDataZero)
  FOAM_MASK_IMAGE_DATA_BOTH(maskImageDataNan)

#define FOAM_MASK_IMAGE_DATA_BOTH_WITH_OUT_IMPL(FUNCTOR, VALUE_TYPE, N_DIM)                                                         \
  m.def(#FUNCTOR,                                                                                                                   \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&, const xt::pytensor<bool, 2>&, VALUE_TYPE, VALUE_TYPE, xt::pytensor<bool, N_DIM>&))  \
    &FUNCTOR<xt::pytensor<VALUE_TYPE, N_DIM>, xt::pytensor<bool, 2>, VALUE_TYPE, xt::pytensor<bool, N_DIM>>,                        \
    py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"), py::arg("out").noconvert());

#define FOAM_MASK_IMAGE_DATA_BOTH_WITH_OUT(FUNCTOR)                                                \
  FOAM_MASK_IMAGE_DATA_BOTH_WITH_OUT_IMPL(FUNCTOR, double, 2)                                      \
  FOAM_MASK_IMAGE_DATA_BOTH_WITH_OUT_IMPL(FUNCTOR, float, 2)

  FOAM_MASK_IMAGE_DATA_BOTH_WITH_OUT(maskImageDataZero)
  FOAM_MASK_IMAGE_DATA_BOTH_WITH_OUT(maskImageDataNan)

  //
  // gain / offset correction
  //

#define FOAM_CORRECT_INTRADARKOFFSET_IMPL(VALUE_TYPE, N_DIM)                                \
  m.def("correctIntraDarkOffset",                                                           \
    (void (*)(xt::pytensor<VALUE_TYPE, N_DIM>&, const xt::pytensor<VALUE_TYPE, N_DIM>&))    \
    &correctIntraDarkImageData<IntraDarkOffsetPolicy, xt::pytensor<VALUE_TYPE, N_DIM>>,              \
    py::arg("src").noconvert(), py::arg("offset").noconvert());

  FOAM_CORRECT_INTRADARKOFFSET_IMPL(double, 3)
  FOAM_CORRECT_INTRADARKOFFSET_IMPL(float, 3)

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
