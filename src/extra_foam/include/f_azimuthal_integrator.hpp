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

template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
auto histogramAI(E&& src, double poni1, double poni2, double pixel1, double pixel2, size_t npt, size_t min_count=1)
{
  auto shape = src.shape();

  // TODO: Separate the geometry and histogram steps. For the same image and integration center, geometry
  //       is only required to calculate once.

  // compute geometry

  xt::xtensor<double, 2> dists = xt::zeros_like(src);
  for (int i = 0; i < static_cast<int>(shape[0]); ++i)
  {
    for (int j = 0; j < static_cast<int>(shape[1]); ++j)
    {
      if (std::isnan(src(i, j))) continue;
      double dx = j * pixel2 - poni2;
      double dy = i * pixel1 - poni1;
      dists(i, j) = std::sqrt(dx * dx + dy * dy);
    }
  }

  // do histogram

  using vector_type = ReducedVectorType<E, double>;

  std::array<double, 2> bounds = xt::minmax(dists)();
  double lb = bounds[0];
  double ub = bounds[1];
  double norm = 1. / (ub - lb);

  vector_type edges = xt::linspace<double>(lb, ub, npt + 1);
  vector_type hist = xt::zeros<double>({ npt });
  xt::xtensor<size_t, 1> counts = xt::zeros<size_t>({ npt });

  for (int i = 0; i < static_cast<int>(shape[0]); ++i)
  {
    for (int j = 0; j < static_cast<int>(shape[1]); ++j)
    {
      double v = dists(i, j);
      if (v == 0.) continue;

      size_t i_bin;
      if (v == ub) i_bin = npt - 1;
      else
        i_bin = static_cast<size_t>(static_cast<double>(npt) * (v - lb) * norm);

      hist(i_bin) += src(i, j);
      counts(i_bin) += 1;
    }
  }

  // thresholding
  if (min_count > 1)
  {
    for (size_t i = 0; i < npt; ++i)
    {
      if (counts(i) < min_count) hist(i) = 0.;
    }
  }

  // normalizing
  for (size_t i = 0; i < npt; ++i)
  {
    if (counts(i) == 0) hist(i) = 0.;
    else
      hist(i) /= counts(i);
  }

  auto&& centers = 0.5 * (xt::view(edges, xt::range(0, -1)) + xt::view(edges, xt::range(1, xt::placeholders::_)));

  return std::make_pair<vector_type, vector_type>(centers, std::move(hist));
}

} //namespace

enum class AzimuthalIntegrationMethod
{
  HISTOGRAM = 0x01,
};


/**
 * class for 1D azimuthal integration of image data.
 */
class AzimuthalIntegrator
{
  double dist_; // sample distance, in m
  xt::xtensor_fixed<double, xt::xshape<3>> poni_; // integration center (y, x, z), in meter
  xt::xtensor_fixed<double, xt::xshape<3>> pixel_; // pixel size (y, x, z), in meter
  double wavelength_; // wavelength, in m

  AzimuthalIntegrationMethod method_;

  /**
   * Convert radial distances to q.
   *
   * q = 4 * pi * sin(theta) / lambda
   *
   * @return: momentum transfer in 1/meter.
   */
  template<typename E>
  void distance2q(E& x) const;

public:

  AzimuthalIntegrator(double dist, double poni1, double poni2, double pixel1, double pixel2, double wavelength);

  ~AzimuthalIntegrator() = default;

  /**
   * Perform 1D azimuthal integration and return the scattering curve.
   *
   * @param src: source image.
   * @param npt: number of integration points.
   * @param min_count: minimum number of pixels required.
   * @param method: azimuthal integration method.
   *
   * @return (q, s): (momentum transfer, scattering)
   */
  template<typename E>
  auto integrate1d(E&& src, size_t npt, size_t min_count=1,
                   AzimuthalIntegrationMethod method=AzimuthalIntegrationMethod::HISTOGRAM) const;
};

template<typename E>
void AzimuthalIntegrator::distance2q(E& x) const
{
  using value_type = typename std::decay_t<E>::value_type;
  x = static_cast<value_type>(4 / wavelength_) *
      static_cast<value_type>(M_PI) / xt::sqrt(4 * dist_ * dist_ / (x * x) + 1);
}

AzimuthalIntegrator::AzimuthalIntegrator(double dist,
                                         double poni1,
                                         double poni2,
                                         double pixel1,
                                         double pixel2,
                                         double wavelength)
  : dist_(dist), poni_({poni1, poni2, 0}), pixel_({pixel1, pixel2, 0}), wavelength_(wavelength)
{
}

template<typename E>
auto AzimuthalIntegrator::integrate1d(E&& src,
                                      size_t npt,
                                      size_t min_count,
                                      AzimuthalIntegrationMethod method) const
{
  if (npt == 0) npt = 1;

  switch(method)
  {
    case AzimuthalIntegrationMethod::HISTOGRAM:
    {
      auto ret = histogramAI(std::forward<E>(src), poni_[0], poni_[1], pixel_[0], pixel_[1], npt, min_count);
      distance2q(ret.first);
      return ret;
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
  double pixel_x_; // pixel size in x direction
  double pixel_y_; // pixel size in y direction

  template<typename E>
  size_t estimateNPoints(const E& src, double cx, double cy) const;

public:

  ConcentricRingsFinder(double pixel_x, double pixel_y);

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
  std::array<double, 2> search(E&& src, double cx0, double cy0, size_t min_count=1) const;

  template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
  auto integrate(const E& src, double cx, double cy, size_t min_count=1) const;
};

ConcentricRingsFinder::ConcentricRingsFinder(double pixel_x, double pixel_y)
  : pixel_x_(pixel_x), pixel_y_(pixel_y)
{
}

template<typename E>
size_t ConcentricRingsFinder::estimateNPoints(const E& src, double cx, double cy) const
{
  auto shape = src.shape();
  auto h = static_cast<double>(shape[0]);
  auto w = static_cast<double>(shape[1]);

  double dx = cx - w;
  double dy = cy - h;
  double max_dist = std::sqrt(cx * cx + cy * cy);
  double dist = std::sqrt(dx * dx + cy * cy);
  if (dist > max_dist) max_dist = dist;
  dist = std::sqrt(cx * cx + dy * dy);
  if (dist > max_dist) max_dist = dist;
  dist = std::sqrt(dx * dx + dy * dy);
  if (dist > max_dist) max_dist = dist;

  return static_cast<size_t>(dist / 2);
}

template<typename E, EnableIf<std::decay_t<E>, IsImage>>
std::array<double, 2> ConcentricRingsFinder::search(E&& src, double cx0, double cy0, size_t min_count) const
{
  double cx_max = cx0;
  double cy_max = cy0;
  double max_s = -1;
  size_t npt = estimateNPoints(src, cx0, cy0);
  using value_type = typename std::decay_t<E>::value_type;

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
          double cx = cx0 + j;
          double cy = cy0 + i;
          double poni1 = cy * pixel_y_;
          double poni2 = cx * pixel_x_;

          auto ret = histogramAI(src, poni1, poni2, pixel_y_, pixel_x_, npt, min_count);

          std::array<double, 2> bounds = xt::minmax(ret.second)();
          double curr_max = bounds[1];

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
auto ConcentricRingsFinder::integrate(const E& src, double cx, double cy, size_t min_count) const
{
  size_t npt = estimateNPoints(src, cx, cy);

  // FIXME: what if pixel x != pixel y
  return histogramAI(src, cy, cx, 1., 1., npt, min_count);
}

} //foam

#endif //EXTRA_FOAM_F_AZIMUTHAL_INTEGRATOR_H
