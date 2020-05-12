/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#ifndef EXTRA_FOAM_F_SMOOTH_H
#define EXTRA_FOAM_F_SMOOTH_H

#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>

#include "f_traits.hpp"
#include "f_utilities.hpp"


namespace foam
{

/**
 * Return the Gaussian kernel.
 *
 * @tparam T: kernel data type.
 * @tparam N: kernel size.
 * @param sigma: Gaussian standard deviation.
 * @return
 */
template<typename T = double>
xt::xtensor<T, 1> getGaussianKernel(size_t k_size, double sigma = -1.)
{
  if (k_size % 2 == 0)
    throw std::invalid_argument("k_size must be odd!");

  // from OpenCV
  sigma = sigma > 0 ? sigma : ((k_size - 1) * 0.5 - 1) * 0.3 + 0.8;

  xt::xtensor<T, 1> x = xt::empty<T>({k_size});
  for (size_t i = 0; i < k_size; ++i) x(i) = i - (k_size - 1) * T(0.5);
  double scale = 0.5 / (sigma * sigma);
  auto ret = xt::exp(-scale * x * x);
  return ret / xt::sum<T>(ret);
}


/**
 * @param Input image.
 *
 * @param src: input image.
 * @param dst: output image of the same size and type as src.
 * @param k_size: Gaussian kernel size.
 * @param sigma: Gaussian kernel standard deviation.
 */
template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
void gaussianBlur(const E& src, E& dst, size_t k_size, double sigma = -1.)
{
  auto shape = src.shape();
  checkShape(shape, dst.shape(), "src and dst have different shapes");

  if (k_size == 0) k_size = 1;

  using value_type = typename std::decay_t<E>::value_type;
  auto kernel = 0.5 * getGaussianKernel<value_type>(k_size, sigma);

  size_t h = shape[0];
  size_t w = shape[1];
  int edge = (static_cast<int>(k_size) - 1) / 2;

  for (size_t i = edge; i < h - edge; ++i)
  {
    for (size_t j = edge; j < w - edge; ++j)
    {
      double sum = 0;
      for (int k = -edge; k <= edge; ++k)
      {
        sum += kernel(k + edge) * src(i + k, j);
      }
      dst(i, j) = sum;
    }
  }

  for (size_t i = edge; i < h - edge; ++i)
  {
    for (size_t j = edge; j < w - edge; ++j)
    {
      double sum = dst(i, j);
      for (int k = -edge; k <= edge; ++k)
      {
        sum += kernel(k + edge) * src(i, j + k);
      }
      dst(i, j) = sum;
    }
  }

  // The pixels on the edges are ignored.
}

} // foam

#endif //EXTRA_FOAM_F_SMOOTH_H
