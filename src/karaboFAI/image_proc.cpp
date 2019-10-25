/**
 * Offline and online data analysis and visualization tool for azimuthal
 * integration of different data acquired with various detectors at
 * European XFEL.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#if defined(FAI_WITH_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#endif

#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include "image_proc.hpp"


namespace fai
{

template<typename T, xt::layout_type L>
struct is_image<xt::pytensor<T, 2, L>> : std::true_type {};

template<typename T, xt::layout_type L>
struct is_image_array<xt::pytensor<T, 3, L>> : std::true_type {};

} // fai

namespace py = pybind11;


PYBIND11_MODULE(image_proc, m)
{
  xt::import_numpy();

  using namespace fai;

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

  // FIXME: nanmeanTwoImages -> nanmeanImageArray when the following bug gets fixed
  // https://github.com/QuantStack/xtensor-python/issues/178
  m.def("nanmeanTwoImages", [] (const xt::pytensor<double, 2>& src1, const xt::pytensor<double, 2>& src2)
    { return nanmeanTwoImages(src1, src2); },
    py::arg("src1").noconvert(), py::arg("src2").noconvert());
  m.def("nanmeanTwoImages", [] (const xt::pytensor<float, 2>& src1, const xt::pytensor<float, 2>& src2)
    { return nanmeanTwoImages(src1, src2); },
    py::arg("src1").noconvert(), py::arg("src2").noconvert());

  m.def("movingAverageImage", &movingAverageImage<xt::pytensor<double, 2>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(),
                              py::arg("count"));
  m.def("movingAverageImage", &movingAverageImage<xt::pytensor<float, 2>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(),
                              py::arg("count"));

  m.def("movingAverageImageArray", &movingAverageImageArray<xt::pytensor<double, 3>>,
                                   py::arg("src").noconvert(), py::arg("data").noconvert(),
                                   py::arg("count"));
  m.def("movingAverageImageArray", &movingAverageImageArray<xt::pytensor<float, 3>>,
                                   py::arg("src").noconvert(), py::arg("data").noconvert(),
                                   py::arg("count"));

  m.def("maskImage", (void (*)(xt::pytensor<double, 2>&, double, double))
                     &maskImage<xt::pytensor<double, 2>, double>,
                     py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskImage", (void (*)(xt::pytensor<float, 2>&, float, float))
                     &maskImage<xt::pytensor<float, 2>, float>,
                     py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskImage", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<bool, 2>&))
                     &maskImage<xt::pytensor<double, 2>, xt::pytensor<bool, 2>>,
                     py::arg("src").noconvert(), py::arg("mask").noconvert());
  m.def("maskImage", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<bool, 2>&))
                     &maskImage<xt::pytensor<float, 2>, xt::pytensor<bool, 2>>,
                     py::arg("src").noconvert(), py::arg("mask").noconvert());

  m.def("maskImage", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<bool, 2>&, double, double))
                     &maskImage<xt::pytensor<double, 2>, xt::pytensor<bool, 2>, double>,
                     py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskImage", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<bool, 2>&, float, float))
                     &maskImage<xt::pytensor<float, 2>, xt::pytensor<bool, 2>, float>,
                     py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskImageArray", &maskImageArray<xt::pytensor<double, 3>, double>,
                          py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskImageArray", &maskImageArray<xt::pytensor<float, 3>, float>,
                          py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskImageArray", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<bool, 2>&))
                          &maskImageArray<xt::pytensor<double, 3>, xt::pytensor<bool, 2>>,
                          py::arg("src").noconvert(), py::arg("mask").noconvert());
  m.def("maskImageArray", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<bool, 2>&))
                          &maskImageArray<xt::pytensor<float, 3>, xt::pytensor<bool, 2>>,
                          py::arg("src").noconvert(), py::arg("mask").noconvert());

  m.def("maskImageArray", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<bool, 2>&, double, double))
                          &maskImageArray<xt::pytensor<double, 3>, xt::pytensor<bool, 2>, double>,
                          py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskImageArray", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<bool, 2>&, float, float))
                          &maskImageArray<xt::pytensor<float, 3>, xt::pytensor<bool, 2>, float>,
                          py::arg("src").noconvert(), py::arg("mask").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("nanToZeroImage", &nanToZeroImage<xt::pytensor<double, 2>>,
                          py::arg("src").noconvert());
  m.def("nanToZeroImage", &nanToZeroImage<xt::pytensor<float, 2>>,
                          py::arg("src").noconvert());

  m.def("nanToZeroImageArray", &nanToZeroImageArray<xt::pytensor<double, 3>>,
                               py::arg("src").noconvert());
  m.def("nanToZeroImageArray", &nanToZeroImageArray<xt::pytensor<float, 3>>,
                               py::arg("src").noconvert());
}
