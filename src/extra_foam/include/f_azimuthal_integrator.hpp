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
#include <xtensor/xsort.hpp>

#include "f_traits.hpp"


namespace foam
{

namespace
{

template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
auto histogramAI(E&& src, double poni1, double poni2, double pixel1, double pixel2, size_t npt, size_t min_count=1)
{
  using value_type = std::conditional_t<std::is_floating_point<typename std::decay_t<E>::value_type>::value,
                                        typename std::decay_t<E>::value_type,
                                        double>;
  using vector_type = ReducedVectorType<E, value_type>;

  auto shape = src.shape();

  vector_type dists = xt::zeros<value_type>({src.size()});
  vector_type weights = xt::zeros<value_type>({src.size()});

  size_t index = 0;
  for (int i = 0; i < static_cast<int>(shape[0]); ++i)
  {
    for (int j = 0; j < static_cast<int>(shape[1]); ++j)
    {
      if (std::isnan(src(i, j))) continue;
      value_type dx = j * pixel2 - poni2;
      value_type dy = i * pixel1 - poni1;
      dists(index) = std::sqrt(dx * dx + dy * dy);
      weights(index++) = src(i, j);
    }
  }

  // do histogram

  std::array<value_type, 2> bounds;
  bounds = xt::minmax(dists)();
  value_type lb = bounds[0];
  value_type ub = bounds[1];
  vector_type edges = xt::linspace<value_type>(lb, ub, npt + 1);
  vector_type hist = xt::zeros<value_type>({ npt });
  vector_type counts = xt::zeros<value_type>({ npt });

  value_type norm = 1. / (ub - lb);
  for (size_t i = 0; i < dists.size(); ++i)
  {
    auto v = dists(i);
    if (v == 0) continue;

    value_type frac = (v - lb) * norm;
    size_t i_bin = npt * frac;
    hist(i_bin) += weights(i);
    counts(i_bin) += 1;
  }

  // thresholding

  if (min_count > 1)
  {
    for (size_t i = 0; i < npt; ++i)
    {
      if (counts(i) < min_count) hist(i) = 0;
    }
  }

  auto&& centers = 0.5 * (xt::view(edges, xt::range(0, -1)) + xt::view(edges, xt::range(1, xt::placeholders::_)));

  // normalizing

  for (size_t i = 0; i < npt; ++i)
  {
    if (counts(i) == 0) hist(i) = 0;
    else
      hist(i) /= counts(i);
  }

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
   * Convert radials to q.
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
  value_type c = 4 / wavelength_;
  x = c * static_cast<value_type>(M_PI) / xt::sqrt(4 * dist_ * dist_ / (x * x) + 1);
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
class ConcentricRingFinder
{
  double pixel_x_; // pixel size in x direction
  double pixel_y_; // pixel size in y direction

public:

  ConcentricRingFinder(double pixel_x, double pixel_y);

  ~ConcentricRingFinder() = default;

  /**
   * Search for the center of concentric rings in an image.
   *
   * @param src: source image.
   * @param cx0: starting x position, in pixels.
   * @param cy0: starting y position, in pixels.
   * @param n_grids: number of grids in searching.
   * @param min_count: minimum number of pixels required for each grid.
   *
   * @return: the best (cx, cy) position in pixels.
   */
  template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
  std::array<double, 2> search(E&& src, double cx0, double cy0,
                               size_t n_grids=128, size_t min_count=1) const;
};

ConcentricRingFinder::ConcentricRingFinder(double pixel_x, double pixel_y)
  : pixel_x_(pixel_x), pixel_y_(pixel_y)
{
}

template<typename E, EnableIf<std::decay_t<E>, IsImage>>
std::array<double, 2> ConcentricRingFinder::search(E&& src, double cx0, double cy0,
                                                   size_t n_grids, size_t min_count) const
{
  if (n_grids == 0) n_grids = 1;

  double cx_max = cx0;
  double cy_max = cy0;
  double max_s = -1;

  int initial_space = 5;
  tbb::mutex mtx;
#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(0, initial_space),
    [&src, cx0, cy0, n_grids, min_count, &cx_max, &cy_max, &max_s, initial_space, &mtx, this]
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

          auto ret = histogramAI(std::forward<E>(src), poni1, poni2, pixel_y_, pixel_x_, n_grids, 10);
          // strangely xt::minmax does support pytensor
          std::array<double, 2> bounds = xt::minmax(xt::xtensor<double, 1>(ret.second))();
          double curr_max = bounds[1];

          {
            tbb::mutex::scoped_lock lock(mtx);
            if (curr_max > max_s)
            {
              max_s = curr_max;
              cx_max = cx;
              cy_max = cy;
            }
          }

        }
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif

  return {cx_max, cy_max};
}

} //foam

#endif //EXTRA_FOAM_F_AZIMUTHAL_INTEGRATOR_H
