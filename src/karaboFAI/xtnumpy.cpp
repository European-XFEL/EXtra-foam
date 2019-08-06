/**
 * Offline and online data analysis and visualization tool for azimuthal
 * integration of different data acquired with various detectors at
 * European XFEL.
 *
 * Numpy functions implemented in C++ (xtensor-python).
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


namespace detail {
  template<typename T>
  inline xt::pytensor<T, 2> _nanmeanImagesImp(const xt::pytensor<T, 3>& arr,
                                              const std::vector<size_t>& keep = {}) {
    auto shape = arr.shape();
    auto mean = xt::pytensor<T, 2>({shape[1], shape[2]});
  
  #if defined(FAI_WITH_TBB)
    tbb::parallel_for(tbb::blocked_range2d<int>(0, shape[1], 0, shape[2]),
      [&arr, &keep, &shape, &mean] (const tbb::blocked_range2d<int> &block) {
        for(int j=block.rows().begin(); j != block.rows().end(); ++j) {
          for(int k=block.cols().begin(); k != block.cols().end(); ++k) {
  #else
    for (std::size_t j=0; j < shape[1]; ++j) {
      for (std::size_t k=0; k < shape[2]; ++k) {
  #endif
        T count = 0;
        T sum = 0;
        if (keep.empty()) {
          for (auto i=0; i<shape[0]; ++i) {
            auto v = arr(i, j, k);
            if (! std::isnan(v)) {
              count += T(1);
              sum += v;
            }
          }
        } else {
          for (auto it=keep.begin(); it != keep.end(); ++it) {
            auto v = arr(*it, j, k);
            if (! std::isnan(v)) {
              count += T(1);
              sum += v;
            }
          }
        }
        
        mean(j, k) = sum / count;
      }
    }
  #if defined(FAI_WITH_TBB)
  });
  #endif
    
    return mean;
  }
} // detail

/**
 * Calculate the nanmean of the selected images from an array of images.
 *
 * @param arr: an array of images. shape = (indices, y, x)
 * @param keep: a list of selected indices.
 * @return: the nanmean image. shape = (y, x)
 */
template<typename T>
inline xt::pytensor<T, 2> nanmeanImages(const xt::pytensor<T, 3>& arr, const std::vector<size_t>& keep) {
  if (keep.empty()) throw std::invalid_argument("keep cannot be empty!");
  return detail::_nanmeanImagesImp(arr, keep);
}

/**
 * Calculate the nanmean of an array of images.
 *
 * @param arr: an array of images. shape = (indices, y, x)
 * @return: the nanmean image. shape = (y, x)
 */
template<typename T>
inline xt::pytensor<T, 2> nanmeanImages(const xt::pytensor<T, 3>& arr) {
  return detail::_nanmeanImagesImp(arr);
}

/**
 * Calculate the nanmean of an array of images.
 *
 * @param arr: an array of images. shape = (indices, y, x)
 * @return: the nanmean image. shape = (y, x)
 */
template<typename T>
inline xt::pytensor<T, 2> xtNanmeanImages(const xt::pytensor<T, 3>& arr) {
  return xt::nanmean<T>(arr, {0}, xt::evaluation_strategy::immediate);
}

// this is even faster than the parallel version of nanmeanImages for np.float32
template<typename T>
inline T nanmeanScalar(T x, T y) {
  if (std::isnan(x) and std::isnan(y)) return 0;

  if (std::isnan(x)) return y;

  if (std::isnan(y)) return x;

  return T(0.5) * (x + y);
}


template<typename T>
inline xt::pyarray<T> movingAverage(xt::pyarray<T>& ma, xt::pyarray<T>& data, size_t count) {
  return ma + (data - ma) / T(count);
}


namespace py = pybind11;


PYBIND11_MODULE(xtnumpy, m) {

  xt::import_numpy();

  m.doc() = "Calculate the mean of images, ignoring NaNs.";

  // only works for float since np.nan is a float?
  m.def("nanmeanImages",
    (xt::pytensor<double, 2> (*)(const xt::pytensor<double, 3>&)) &nanmeanImages<double>);
  m.def("nanmeanImages",
    (xt::pytensor<float, 2> (*)(const xt::pytensor<float, 3>&)) &nanmeanImages<float>);
  m.def("nanmeanImages",
    (xt::pytensor<double, 2> (*)(const xt::pytensor<double, 3>&, const std::vector<size_t>&))
    &nanmeanImages<double>);
  m.def("nanmeanImages",
    (xt::pytensor<float, 2> (*)(const xt::pytensor<float, 3>&, const std::vector<size_t>&))
    &nanmeanImages<float>);
  m.def("xtNanmeanImages", &xtNanmeanImages<double>);
  m.def("xtNanmeanImages", &xtNanmeanImages<float>);
  m.def("xt_nanmean_two_images", xt::pyvectorize(nanmeanScalar<double>));
  m.def("xt_nanmean_two_images", xt::pyvectorize(nanmeanScalar<float>));
  m.def("xt_moving_average", &movingAverage<double>);
  m.def("xt_moving_average", &movingAverage<float>);
}
