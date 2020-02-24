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

  m.doc() = "Calculate the mean of images, ignoring NaNs.";

  m.def("nanmeanImageArray", [] (const xt::pytensor<double, 3>& src)
    { return nanmeanImageArray(src); },
    py::arg("src").noconvert());
  m.def("nanmeanImageArray", [] (const xt::pytensor<float, 3>& src)
    { return nanmeanImageArray(src); },
    py::arg("src").noconvert());

  m.def("nanmeanImageArray", [] (const xt::pytensor<double, 3>& src, const std::vector<size_t>& keep)
    { return nanmeanImageArray(src, keep); },
    py::arg("src").noconvert(), py::arg("keep"));
  m.def("nanmeanImageArray", [] (const xt::pytensor<float, 3>& src, const std::vector<size_t>& keep)
    { return nanmeanImageArray(src, keep); },
    py::arg("src").noconvert(), py::arg("keep"));

  m.def("nanmeanImageArray", [] (const xt::pytensor<double, 2>& src1, const xt::pytensor<double, 2>& src2)
    { return nanmeanImageArray(src1, src2); },
    py::arg("src1").noconvert(), py::arg("src2").noconvert());
  m.def("nanmeanImageArray", [] (const xt::pytensor<float, 2>& src1, const xt::pytensor<float, 2>& src2)
    { return nanmeanImageArray(src1, src2); },
    py::arg("src1").noconvert(), py::arg("src2").noconvert());

  m.def("movingAvgImageData", &movingAvgImageData<xt::pytensor<double, 2>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(),
                              py::arg("count"));
  m.def("movingAvgImageData", &movingAvgImageData<xt::pytensor<float, 2>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(),
                              py::arg("count"));

  m.def("movingAvgImageData", &movingAvgImageData<xt::pytensor<double, 3>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(),
                              py::arg("count"));
  m.def("movingAvgImageData", &movingAvgImageData<xt::pytensor<float, 3>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(),
                              py::arg("count"));

  m.def("maskZeroImageData", &maskZeroImageData<xt::pytensor<double, 2>>, py::arg("src").noconvert());
  m.def("maskZeroImageData", &maskZeroImageData<xt::pytensor<float, 2>>, py::arg("src").noconvert());

  m.def("maskZeroImageData", (void (*)(xt::pytensor<double, 2>&, double, double))
                             &maskZeroImageData<xt::pytensor<double, 2>, double>,
                             py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskZeroImageData", (void (*)(xt::pytensor<float, 2>&, float, float))
                             &maskZeroImageData<xt::pytensor<float, 2>, float>,
                             py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskZeroImageData", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<bool, 2>&))
                             &maskZeroImageData<xt::pytensor<double, 2>, xt::pytensor<bool, 2>>,
                             py::arg("src").noconvert(), py::arg("mask").noconvert());
  m.def("maskZeroImageData", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<bool, 2>&))
                             &maskZeroImageData<xt::pytensor<float, 2>, xt::pytensor<bool, 2>>,
                             py::arg("src").noconvert(), py::arg("mask").noconvert());

  m.def("maskZeroImageData", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<bool, 2>&, double, double))
                             &maskZeroImageData<xt::pytensor<double, 2>, xt::pytensor<bool, 2>, double>,
                             py::arg("src").noconvert(), py::arg("mask").noconvert(),
                             py::arg("lb"), py::arg("ub"));
  m.def("maskZeroImageData", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<bool, 2>&, float, float))
                             &maskZeroImageData<xt::pytensor<float, 2>, xt::pytensor<bool, 2>, float>,
                             py::arg("src").noconvert(), py::arg("mask").noconvert(),
                             py::arg("lb"), py::arg("ub"));

  m.def("maskZeroImageData", &maskZeroImageData<xt::pytensor<double, 3>>, py::arg("src").noconvert());
  m.def("maskZeroImageData", &maskZeroImageData<xt::pytensor<float, 3>>, py::arg("src").noconvert());

  m.def("maskZeroImageData", &maskZeroImageData<xt::pytensor<double, 3>, double>,
                             py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskZeroImageData", &maskZeroImageData<xt::pytensor<float, 3>, float>,
                             py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskZeroImageData", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<bool, 2>&))
                             &maskZeroImageData<xt::pytensor<double, 3>, xt::pytensor<bool, 2>>,
                             py::arg("src").noconvert(), py::arg("mask").noconvert());
  m.def("maskZeroImageData", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<bool, 2>&))
                             &maskZeroImageData<xt::pytensor<float, 3>, xt::pytensor<bool, 2>>,
                             py::arg("src").noconvert(), py::arg("mask").noconvert());

  m.def("maskZeroImageData", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<bool, 2>&, double, double))
                             &maskZeroImageData<xt::pytensor<double, 3>, xt::pytensor<bool, 2>, double>,
                             py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskZeroImageData", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<bool, 2>&, float, float))
                             &maskZeroImageData<xt::pytensor<float, 3>, xt::pytensor<bool, 2>, float>,
                             py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskNanImageData", &maskNanImageData<xt::pytensor<double, 2>>, py::arg("src").noconvert());
  m.def("maskNanImageData", &maskNanImageData<xt::pytensor<float, 2>>, py::arg("src").noconvert());

  m.def("maskNanImageData", (void (*)(xt::pytensor<double, 2>&, double, double))
                            &maskNanImageData<xt::pytensor<double, 2>, double>,
                            py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskNanImageData", (void (*)(xt::pytensor<float, 2>&, float, float))
                            &maskNanImageData<xt::pytensor<float, 2>, float>,
                            py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskNanImageData", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<bool, 2>&))
                            &maskNanImageData<xt::pytensor<double, 2>, xt::pytensor<bool, 2>>,
                            py::arg("src").noconvert(), py::arg("mask").noconvert());
  m.def("maskNanImageData", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<bool, 2>&))
                            &maskNanImageData<xt::pytensor<float, 2>, xt::pytensor<bool, 2>>,
                            py::arg("src").noconvert(), py::arg("mask").noconvert());

  m.def("maskNanImageData", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<bool, 2>&, double, double))
                            &maskNanImageData<xt::pytensor<double, 2>, xt::pytensor<bool, 2>, double>,
                            py::arg("src").noconvert(), py::arg("mask").noconvert(),
                            py::arg("lb"), py::arg("ub"));
  m.def("maskNanImageData", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<bool, 2>&, float, float))
                            &maskNanImageData<xt::pytensor<float, 2>, xt::pytensor<bool, 2>, float>,
                            py::arg("src").noconvert(), py::arg("mask").noconvert(),
                            py::arg("lb"), py::arg("ub"));

  m.def("maskNanImageData", &maskNanImageData<xt::pytensor<double, 3>>, py::arg("src").noconvert());
  m.def("maskNanImageData", &maskNanImageData<xt::pytensor<float, 3>>, py::arg("src").noconvert());

  m.def("maskNanImageData", &maskNanImageData<xt::pytensor<double, 3>, double>,
                            py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskNanImageData", &maskNanImageData<xt::pytensor<float, 3>, float>,
                            py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskNanImageData", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<bool, 2>&))
                            &maskNanImageData<xt::pytensor<double, 3>, xt::pytensor<bool, 2>>,
                            py::arg("src").noconvert(), py::arg("mask").noconvert());
  m.def("maskNanImageData", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<bool, 2>&))
                            &maskNanImageData<xt::pytensor<float, 3>, xt::pytensor<bool, 2>>,
                            py::arg("src").noconvert(), py::arg("mask").noconvert());

  m.def("maskNanImageData", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<bool, 2>&, double, double))
                            &maskNanImageData<xt::pytensor<double, 3>, xt::pytensor<bool, 2>, double>,
                            py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskNanImageData", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<bool, 2>&, float, float))
                            &maskNanImageData<xt::pytensor<float, 3>, xt::pytensor<bool, 2>, float>,
                            py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("correctOffset", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<double, 3>&))
                         &correctImageData<OffsetPolicy, xt::pytensor<double, 3>>,
                         py::arg("src").noconvert(), py::arg("offset").noconvert());
  m.def("correctOffset", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<float, 3>&))
                         &correctImageData<OffsetPolicy, xt::pytensor<float, 3>>,
                         py::arg("src").noconvert(), py::arg("offset").noconvert());

  m.def("correctOffset", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<double, 2>&))
                         &correctImageData<OffsetPolicy, xt::pytensor<double, 2>>,
                         py::arg("src").noconvert(), py::arg("offset").noconvert());
  m.def("correctOffset", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<float, 2>&))
                         &correctImageData<OffsetPolicy, xt::pytensor<float, 2>>,
                         py::arg("src").noconvert(), py::arg("offset").noconvert());

  m.def("correctGain", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<double, 3>&))
                       &correctImageData<GainPolicy, xt::pytensor<double, 3>>,
                       py::arg("src").noconvert(), py::arg("gain").noconvert());
  m.def("correctGain", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<float, 3>&))
                       &correctImageData<GainPolicy, xt::pytensor<float, 3>>,
                       py::arg("src").noconvert(), py::arg("gain").noconvert());

  m.def("correctGain", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<double, 2>&))
                       &correctImageData<GainPolicy, xt::pytensor<double, 2>>,
                       py::arg("src").noconvert(), py::arg("gain").noconvert());
  m.def("correctGain", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<float, 2>&))
                       &correctImageData<GainPolicy, xt::pytensor<float, 2>>,
                       py::arg("src").noconvert(), py::arg("gain").noconvert());

  m.def("correctGainOffset", (void (*)(xt::pytensor<double, 3>&,
                                       const xt::pytensor<double, 3>&, const xt::pytensor<double, 3>&))
                             &correctImageData<xt::pytensor<double, 3>>,
                             py::arg("src").noconvert(), py::arg("gain").noconvert(), py::arg("offset").noconvert());
  m.def("correctGainOffset", (void (*)(xt::pytensor<float, 3>&,
                                       const xt::pytensor<float, 3>&, const xt::pytensor<float, 3>&))
                             &correctImageData<xt::pytensor<float, 3>>,
                             py::arg("src").noconvert(), py::arg("gain").noconvert(), py::arg("offset").noconvert());

  m.def("correctGainOffset", (void (*)(xt::pytensor<double, 2>&,
                                       const xt::pytensor<double, 2>&, const xt::pytensor<double, 2>&))
                             &correctImageData<xt::pytensor<double, 2>>,
                             py::arg("src").noconvert(), py::arg("gain").noconvert(), py::arg("offset").noconvert());
  m.def("correctGainOffset", (void (*)(xt::pytensor<float, 2>&,
                                       const xt::pytensor<float, 2>&, const xt::pytensor<float, 2>&))
                             &correctImageData<xt::pytensor<float, 2>>,
                             py::arg("src").noconvert(), py::arg("gain").noconvert(), py::arg("offset").noconvert());
}
