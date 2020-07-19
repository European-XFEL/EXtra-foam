/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#ifndef EXTRA_FOAM_F_AZIMUTHAL_INTEGRATOR_H
#define EXTRA_FOAM_F_AZIMUTHAL_INTEGRATOR_H

#include <cmath>

#if defined(FOAM_USE_TBB)
#include "tbb/parallel_for.h"
#include "tbb/mutex.h"
#endif

#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>

#include "f_traits.hpp"


namespace foam
{

namespace
{

template<typename T, typename E, EnableIf<std::decay_t<E>, IsImage> = false>
auto computeGeometry(E&& src, T poni1, T poni2, T pixel1, T pixel2)
{
  auto shape = src.shape();

  xt::xtensor<T, 2> geometry = xt::zeros<T>(shape);
  for (size_t i = 0; i < shape[0]; ++i)
  {
    for (size_t j = 0; j < shape[1]; ++j)
    {
      T dx = static_cast<T>(j) * pixel2 - poni2;
      T dy = static_cast<T>(i) * pixel1 - poni1;
      geometry(i, j) = std::sqrt(dx * dx + dy * dy);
    }
  }

  return geometry;
}

template<typename T, typename E, EnableIf<std::decay_t<E>, IsImage> = false>
auto histogramAI(E&& src, const xt::xtensor<T, 2>& geometry, T q_min, T q_max,
                 size_t n_bins, size_t min_count=1)
{
  auto shape = src.shape();

  using vector_type = ReducedVectorType<E, T>;

  T norm = T(1) / (q_max - q_min);

  vector_type edges = xt::linspace<T>(q_min, q_max, n_bins + 1);
  vector_type hist = xt::zeros<T>({ n_bins });
  xt::xtensor<size_t, 1> counts = xt::zeros<size_t>({ n_bins });

  for (size_t i = 0; i < shape[0]; ++i)
  {
    for (size_t j = 0; j < shape[1]; ++j)
    {
      T q = geometry(i, j);
      auto v = src(i, j);

      if (std::isnan(v)) continue;

      if (q == q_max)
      {
        hist(n_bins - 1) += v;
        counts(n_bins - 1) += 1;
      } else if ( (q > q_min) && (q < q_max) )
      {
        auto i_bin = static_cast<size_t>(static_cast<T>(n_bins) * (q - q_min) * norm);
        hist(i_bin) += v;
        counts(i_bin) += 1;
      }
    }
  }

  // thresholding
  if (min_count > 1)
  {
    for (size_t i = 0; i < n_bins; ++i)
    {
      if (counts(i) < min_count) hist(i) = 0.;
    }
  }

  // normalizing
  for (size_t i = 0; i < n_bins; ++i)
  {
    if (counts(i) == 0) hist(i) = 0.;
    else
      hist(i) /= counts(i);
  }

  auto&& centers = 0.5 * (xt::view(edges, xt::range(0, -1)) + xt::view(edges, xt::range(1, xt::placeholders::_)));

  return std::make_pair<vector_type, vector_type>(centers, std::move(hist));
}

template<typename T, typename E, EnableIf<std::decay_t<E>, IsImage> = false>
auto histogramAI(E&& src, T poni1, T poni2, T pixel1, T pixel2, size_t npt, size_t min_count=1)
{
  xt::xtensor<T, 2> geometry = computeGeometry(src, poni1, poni2, pixel1, pixel2);

  std::array<T, 2> bounds = xt::minmax(geometry)();

  return histogramAI<T>(std::forward<E>(src), geometry, bounds[0], bounds[1], npt, min_count);
}

} //namespace

enum class AzimuthalIntegrationMethod
{
  HISTOGRAM = 0x01,
};


/**
 * class for 1D azimuthal integration of image data.
 */
template<typename T = double>
class AzimuthalIntegrator
{
public:

  using value_type = std::conditional_t<std::is_floating_point<T>::value, T, double>;

private:

  value_type dist_; // sample distance, in m
  xt::xtensor_fixed<value_type, xt::xshape<3>> poni_; // integration center (y, x, z), in meter
  xt::xtensor_fixed<value_type, xt::xshape<3>> pixel_; // pixel size (y, x, z), in meter
  value_type wavelength_; // wavelength, in m

  bool initialized_ = false;
  xt::xtensor<value_type, 2> q_;
  value_type q_min_;
  value_type q_max_;

  AzimuthalIntegrationMethod method_;

  /**
   * Convert radial distances (in meter) to momentum transfer q (in 1/meter).
   *
   * q = 4 * pi * sin(theta) / lambda
   *
   * @return: momentum transfer in 1/meter.
   */
  template<typename E>
  void distance2q(E& x) const;

  /**
   * Initialize Q-map.
   */
  template<typename E>
  void initQ(const E& src);

public:

  AzimuthalIntegrator(double dist, double poni1, double poni2, double pixel1, double pixel2, double wavelength);

  ~AzimuthalIntegrator() = default;

  /**
   * Perform 1D azimuthal integration for a single image.
   *
   * @param src: source image.
   * @param npt: number of integration points.
   * @param min_count: minimum number of pixels required.
   * @param method: azimuthal integration method.
   *
   * @return (q, s): (momentum transfer, scattering)
   */
  template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
  auto integrate1d(E&& src, size_t npt, size_t min_count=1,
                   AzimuthalIntegrationMethod method=AzimuthalIntegrationMethod::HISTOGRAM);
};

template<typename T>
template<typename E>
void AzimuthalIntegrator<T>::distance2q(E& x) const
{
  x = static_cast<value_type>(value_type(4.) / wavelength_) *
      static_cast<value_type>(M_PI) / xt::sqrt(value_type(4.) * dist_ * dist_ / (x * x) + value_type(1.));
}

template<typename T>
template<typename E>
void AzimuthalIntegrator<T>::initQ(const E& src)
{
  q_ = computeGeometry(src, poni_[0], poni_[1], pixel_[0], pixel_[1]);
  distance2q(q_);
  std::array<value_type, 2> bounds = xt::minmax(q_)();
  q_min_ = bounds[0];
  q_max_ = bounds[1];
}

template<typename T>
AzimuthalIntegrator<T>::AzimuthalIntegrator(double dist,
                                            double poni1,
                                            double poni2,
                                            double pixel1,
                                            double pixel2,
                                            double wavelength)
  : dist_(static_cast<value_type>(dist)),
    poni_({static_cast<value_type>(poni1), static_cast<value_type>(poni2), 0}),
    pixel_({static_cast<value_type>(pixel1), static_cast<value_type>(pixel2), 0}),
    wavelength_(static_cast<value_type>(wavelength))
{
}

template<typename T>
template<typename E, EnableIf<std::decay_t<E>, IsImage>>
auto AzimuthalIntegrator<T>::integrate1d(E&& src,
                                         size_t npt,
                                         size_t min_count,
                                         AzimuthalIntegrationMethod method)
{
  if (npt == 0) npt = 1;

  auto src_shape = src.shape();
  std::array<size_t, 2> q_shape = q_.shape();
  if (!initialized_ || src_shape[0] != q_shape[0] || src_shape[1] != q_shape[1])
  {
    initQ(src);
    initialized_ = true;
  }

  switch(method)
  {
    case AzimuthalIntegrationMethod::HISTOGRAM:
    {
      return histogramAI(std::forward<E>(src), q_, q_min_, q_max_, npt, min_count);
    }
    default:
      throw std::runtime_error("Unknown azimuthal integration method");
  }
}

/**
 * class for finding the center of concentric rings in an image.
 */
class ConcentricRingsFinder
{
  float pixel_x_; // pixel size in x direction
  float pixel_y_; // pixel size in y direction

  template<typename E>
  size_t estimateNPoints(const E& src, float cx, float cy) const;

public:

  ConcentricRingsFinder(float pixel_x, float pixel_y);

  ~ConcentricRingsFinder() = default;

  /**
   * Search for the center of concentric rings in an image.
   *
   * @param src: source image.
   * @param cx0: starting x position, in pixels.
   * @param cy0: starting y position, in pixels.
   * @param min_count: minimum number of pixels required for each grid.
   *
   * @return: the optimized (cx, cy) position in pixels.
   */
  template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
  std::array<float, 2> search(E&& src, float cx0, float cy0, size_t min_count=1) const;

  template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
  auto integrate(E&& src, float cx, float cy, size_t min_count=1) const;
};

ConcentricRingsFinder::ConcentricRingsFinder(float pixel_x, float pixel_y)
  : pixel_x_(pixel_x), pixel_y_(pixel_y)
{
}

template<typename E>
size_t ConcentricRingsFinder::estimateNPoints(const E& src, float cx, float cy) const
{
  auto shape = src.shape();
  auto h = static_cast<float>(shape[0]);
  auto w = static_cast<float>(shape[1]);

  float dx = cx - w;
  float dy = cy - h;
  float max_dist = std::sqrt(cx * cx + cy * cy);
  float dist = std::sqrt(dx * dx + cy * cy);
  if (dist > max_dist) max_dist = dist;
  dist = std::sqrt(cx * cx + dy * dy);
  if (dist > max_dist) max_dist = dist;
  dist = std::sqrt(dx * dx + dy * dy);
  if (dist > max_dist) max_dist = dist;

  return static_cast<size_t>(dist / 2);
}

template<typename E, EnableIf<std::decay_t<E>, IsImage>>
std::array<float, 2> ConcentricRingsFinder::search(E&& src, float cx0, float cy0, size_t min_count) const
{
  float cx_max = cx0;
  float cy_max = cy0;
  float max_s = -1;
  size_t npt = estimateNPoints(src, cx0, cy0);

  int initial_space = 10;
#if defined(FOAM_USE_TBB)
  tbb::mutex mtx;
  tbb::parallel_for(tbb::blocked_range<int>(-initial_space, initial_space),
    [&src, cx0, cy0, npt, min_count, &cx_max, &cy_max, &max_s, initial_space, &mtx, this]
    (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
#else
      for (int i = -initial_space; i <= initial_space; ++i)
      {
#endif
        for (int j = -initial_space; j <= initial_space; ++j)
        {
          float cx = cx0 + j;
          float cy = cy0 + i;
          float poni1 = cy * pixel_y_;
          float poni2 = cx * pixel_x_;

          auto ret = histogramAI<float>(src, poni1, poni2, pixel_y_, pixel_x_, npt, min_count);

          std::array<float, 2> bounds = xt::minmax(ret.second)();
          float curr_max = bounds[1];

#if defined(FOAM_USE_TBB)
          {
            tbb::mutex::scoped_lock lock(mtx);
#endif
            if (curr_max > max_s)
            {
              max_s = curr_max;
              cx_max = cx;
              cy_max = cy;
            }
#if defined(FOAM_USE_TBB)
          }
#endif
        }
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif

  return {cx_max, cy_max};
}

template<typename E, EnableIf<std::decay_t<E>, IsImage>>
auto ConcentricRingsFinder::integrate(E&& src, float cx, float cy, size_t min_count) const
{
  size_t npt = estimateNPoints(src, cx, cy);

  // FIXME: what if pixel x != pixel y
  return histogramAI<float>(std::forward<E>(src), cy, cx, 1., 1., npt, min_count);
}

} //foam

#endif //EXTRA_FOAM_F_AZIMUTHAL_INTEGRATOR_H
