/**
 * Offline and online data analysis and visualization tool for azimuthal
 * integration of different data acquired with various detectors at
 * European XFEL.
 *
 * Numpy functions implemented in C++.
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
namespace detail
{

template<typename E, template <typename> class C = is_train,
  check_container<std::decay_t<E>, C> = false>
inline auto nanmeanImagesImp(E&& src, const std::vector<size_t>& keep = {})
{
  using value_type = typename std::decay_t<E>::value_type;
  using container_type = typename std::decay_t<E>::base_type;
  auto shape = src.shape();

  // TODO: deduce result type
  auto mean = xt::pytensor<value_type, 2>({shape[1], shape[2]});

#if defined(FAI_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range2d<int>(0, shape[1], 0, shape[2]),
    [&src, &keep, &shape, &mean] (const tbb::blocked_range2d<int> &block)
    {
      for(int j=block.rows().begin(); j != block.rows().end(); ++j)
      {
        for(int k=block.cols().begin(); k != block.cols().end(); ++k)
        {
#else
      for (std::size_t j=0; j < shape[1]; ++j)
      {
        for (std::size_t k=0; k < shape[2]; ++k)
        {
#endif
          std::size_t count = 0;
          value_type sum = 0;
          if (keep.empty())
          {
            for (auto i=0; i<shape[0]; ++i)
            {
              auto v = src(i, j, k);
              if (! std::isnan(v))
              {
                count += 1;
                sum += v;
              }
            }
          }
          else
          {
            for (auto it=keep.begin(); it != keep.end(); ++it)
            {
              auto v = src(*it, j, k);
              if (! std::isnan(v))
              {
                count += 1;
                sum += v;
              }
            }
          }

          mean(j, k) = sum / value_type(count);
        }
      }
#if defined(FAI_WITH_TBB)
    }
  );
#endif
  return mean;
}

} // detail

template<typename T, xt::layout_type L>
struct is_pulse<xt::pytensor<T, 2, L>> : std::true_type {};

template<typename T, xt::layout_type L>
struct is_train<xt::pytensor<T, 3, L>> : std::true_type {};

/**
 * Calculate the nanmean of the selected images from an array of images.
 *
 * @param src: image data. shape = (indices, y, x)
 * @param keep: a list of selected indices.
 * @return: the nanmean image. shape = (y, x)
 */
template<typename E>
inline auto nanmeanTrain(E&& src, const std::vector<size_t>& keep)
{
  if (keep.empty()) throw std::invalid_argument("keep cannot be empty!");
  return detail::nanmeanImagesImp(std::forward<E>(src), keep);
}

/**
 * Calculate the nanmean of an array of images.
 *
 * @param src: image data. shape = (indices, y, x)
 * @return: the nanmean image. shape = (y, x)
 */
template<typename E>
inline auto nanmeanTrain(E&& src)
{
  return detail::nanmeanImagesImp(std::forward<E>(src));
}

} // fai

namespace py = pybind11;


PYBIND11_MODULE(xtnumpy, m)
{
  xt::import_numpy();

  using namespace fai;

  m.doc() = "Calculate the mean of images, ignoring NaNs.";

  m.def("nanmeanTrain", [] (const xt::pytensor<double, 3>& src) { return nanmeanTrain(src); },
                        py::arg("src").noconvert());
  m.def("nanmeanTrain", [] (const xt::pytensor<float, 3>& src) { return nanmeanTrain(src); },
                        py::arg("src").noconvert());

  m.def("nanmeanTrain", [] (const xt::pytensor<double, 3>& src, const std::vector<size_t>& keep)
    { return nanmeanTrain(src, keep); }, py::arg("src").noconvert(), py::arg("keep"));
  m.def("nanmeanTrain", [] (const xt::pytensor<float, 3>& src, const std::vector<size_t>& keep)
    { return nanmeanTrain(src, keep); }, py::arg("src").noconvert(), py::arg("keep"));

  // FIXME: nanmeanTwo -> nanmeanImage when the following bug gets fixed
  // https://github.com/QuantStack/xtensor-python/issues/178
  m.def("nanmeanTwo", [] (const xt::pytensor<double, 2>& src1, const xt::pytensor<double, 2>& src2)
    { return nanmeanTrain(src1, src2); }, py::arg("src1").noconvert(), py::arg("src2").noconvert());
  m.def("nanmeanTwo", [] (const xt::pytensor<float, 2>& src1, const xt::pytensor<float, 2>& src2)
    { return nanmeanTrain(src1, src2); }, py::arg("src1").noconvert(), py::arg("src2").noconvert());

  m.def("xtNanmeanTrain", [] (const xt::pytensor<double, 3>& src) -> xt::pytensor<double, 2>
                          { return xtNanmeanTrain(src); }, py::arg("src").noconvert());
  m.def("xtNanmeanTrain", [] (const xt::pytensor<float, 3>& src) -> xt::pytensor<float, 2>
                          { return xtNanmeanTrain(src); }, py::arg("src").noconvert());

  m.def("movingAveragePulse", &movingAveragePulse<xt::pytensor<double, 2>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(), py::arg("count"));
  m.def("movingAveragePulse", &movingAveragePulse<xt::pytensor<float, 2>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(), py::arg("count"));

  m.def("movingAverageTrain", &movingAverageTrain<xt::pytensor<double, 3>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(), py::arg("count"));
  m.def("movingAverageTrain", &movingAverageTrain<xt::pytensor<float, 3>>,
                              py::arg("src").noconvert(), py::arg("data").noconvert(), py::arg("count"));

  m.def("maskPulse", (void (*)(xt::pytensor<double, 2>&, double, double))
                     &maskPulse<xt::pytensor<double, 2>, double>,
                     py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskPulse", (void (*)(xt::pytensor<float, 2>&, float, float))
                     &maskPulse<xt::pytensor<float, 2>, float>,
                     py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskPulse", (void (*)(xt::pytensor<double, 2>&, const xt::pytensor<bool, 2>&))
                     &maskPulse<xt::pytensor<double, 2>, xt::pytensor<bool, 2>>,
                     py::arg("src").noconvert(), py::arg("mask").noconvert());
  m.def("maskPulse", (void (*)(xt::pytensor<float, 2>&, const xt::pytensor<bool, 2>&))
                     &maskPulse<xt::pytensor<float, 2>, xt::pytensor<bool, 2>>,
                     py::arg("src").noconvert(), py::arg("mask").noconvert());

  m.def("maskTrain", &maskTrain<xt::pytensor<double, 3>, double>,
                     py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("maskTrain", &maskTrain<xt::pytensor<float, 3>, float>,
                     py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("maskTrain", (void (*)(xt::pytensor<double, 3>&, const xt::pytensor<bool, 2>&))
                     &maskTrain<xt::pytensor<double, 3>, xt::pytensor<bool, 2>>,
                     py::arg("src").noconvert(), py::arg("mask").noconvert());
  m.def("maskTrain", (void (*)(xt::pytensor<float, 3>&, const xt::pytensor<bool, 2>&))
                     &maskTrain<xt::pytensor<float, 3>, xt::pytensor<bool, 2>>,
                     py::arg("src").noconvert(), py::arg("mask").noconvert());

  m.def("xtMaskTrain", &xtMaskTrain<xt::pytensor<double, 3>, double>,
                       py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));
  m.def("xtMaskTrain", &xtMaskTrain<xt::pytensor<float, 3>, float>,
                       py::arg("src").noconvert(), py::arg("lb"), py::arg("ub"));

  m.def("nanToZeroPulse", &nanToZeroPulse<xt::pytensor<double, 2>>, py::arg("src").noconvert());
  m.def("nanToZeroPulse", &nanToZeroPulse<xt::pytensor<float, 2>>, py::arg("src").noconvert());

  m.def("nanToZeroTrain", &nanToZeroTrain<xt::pytensor<double, 3>>, py::arg("src").noconvert());
  m.def("nanToZeroTrain", &nanToZeroTrain<xt::pytensor<float, 3>>, py::arg("src").noconvert());
}
