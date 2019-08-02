/**
 * Offline and online data analysis and visualization tool for azimuthal
 * integration of different data acquired with various detectors at
 * European XFEL.
 *
 * Numpy functions implemented in xtensor.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#include <iostream>

#include "pybind11/pybind11.h"

#if defined(FAI_WITH_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#endif

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"


template<typename T>
inline xt::pytensor<T, 2> nanmeanImages(const xt::pytensor<T, 3>& arr) {

  auto shape = arr.shape();
  auto mean = xt::pytensor<T, 2>({shape[1], shape[2]});

#if defined(FAI_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range2d<int>(0, shape[1], 0, shape[2]),
    [&arr, &shape, &mean] (const tbb::blocked_range2d<int> &block) {
      for(int j=block.rows().begin(); j != block.rows().end(); ++j){
        for(int k=block.cols().begin(); k != block.cols().end(); ++k){
#else
  for (std::size_t j=0; j < shape[1]; ++j) {
    for (std::size_t k=0; k < shape[2]; ++k) {
#endif
          T count = 0;
          T sum = 0;
          for (std::size_t i=0; i < shape[0]; ++i) {
            auto v = arr(i, j, k);
            if (! std::isnan(v)) {
              count += T(1);
              sum += v;
            }
          }
          mean(j, k) = sum / count;
        }
      }
#if defined(FAI_WITH_TBB)
    }
  );
#endif

  return mean;
}


template<typename T>
inline xt::pytensor<T, 2> nanmeanImagesOld(const xt::pytensor<T, 3>& arr) {
  return xt::nanmean<T>(arr, {0}, xt::evaluation_strategy::immediate);
}


template<typename T>
inline T nanmeanScalar(T x, T y) {
  if (std::isnan(x) and std::isnan(y)) return 0;

  if (std::isnan(x)) return y;

  if (std::isnan(y)) return x;

  return (x + y) / 2;
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
  m.def("xt_nanmean_images", &nanmeanImages<double>);
  m.def("xt_nanmean_images", &nanmeanImages<float>);
  m.def("xt_nanmean_images_old", &nanmeanImagesOld<double>);
  m.def("xt_nanmean_images_old", &nanmeanImagesOld<float>);
  m.def("xt_nanmean_two_images", xt::pyvectorize(nanmeanScalar<double>));
  m.def("xt_nanmean_two_images", xt::pyvectorize(nanmeanScalar<float>));
  m.def("xt_moving_average", &movingAverage<double>);
  m.def("xt_moving_average", &movingAverage<float>);
}
