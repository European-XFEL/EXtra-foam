/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#ifndef EXTRA_FOAM_F_CANNY_HPP
#define EXTRA_FOAM_F_CANNY_HPP

#include <cmath>

#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>

#include "f_traits.hpp"
#include "f_utilities.hpp"


namespace foam
{

/**
 * Finds edges in an image using the Canny algorithm.
 *
 * @param src: input image.
 * @param dst: output image of the same size as src.
 * @param low_thresh: first threshold for the hysteresis procedure.
 * @param high_thresh: second threshold for the hysteresis procedure.
 */
template<typename E, typename F,
  EnableIf<std::decay_t<E>, IsImage> = false, EnableIf<std::decay_t<F>, IsImage> = false>
void cannyEdge(E&& src, F& dst,
               double lt = std::numeric_limits<double>::min(), double ht = std::numeric_limits<double>::max())
{
  auto shape = src.shape();
  size_t h = shape[0];
  size_t w = shape[1];
  assert(h > 1);
  assert(w > 1);

  if (lt > ht) std::swap(lt, ht);

  using container_type = std::decay_t<E>;
  using value_type = typename container_type::value_type;

  container_type sm = xt::zeros<value_type>({h, w});
  container_type sa = xt::zeros<value_type>({h, w});

  xt::xtensor_fixed<value_type, xt::xshape<3, 3>> gx {
    {-1,  0,  1},
    {-2,  0,  2},
    {-1,  0,  1}
  };

  xt::xtensor_fixed<value_type, xt::xshape<3, 3>> gy {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
  };

  // computing gradient and angle
  for (size_t i = 1; i < h - 1; ++i)
  {
    for (size_t j = 1; j < w - 1; ++j)
    {
      auto x = src(i-1, j-1) * gx(0, 0) + src(i, j-1) * gx(1, 0) + src(i+1, j-1) * gx(2, 0)
               + src(i-1, j) * gx(0, 1) + src(i, j) * gx(1, 1) + src(i+1, j) * gx(2, 1)
               + src(i-1, j+1) * gx(0, 2) + src(i, j+1) * gx(1, 2) + src(i+1, j+1) * gx(2, 2);

      auto y = src(i-1, j-1) * gy(0, 0) + src(i, j-1) * gy(1, 0) + src(i+1, j-1) * gy(2, 0)
               + src(i-1, j) * gy(0, 1) + src(i, j) * gy(1, 1) + src(i+1, j) * gy(2, 1)
               + src(i-1, j+1) * gy(0, 2) + src(i, j+1) * gy(1, 2) + src(i+1, j+1) * gy(2, 2);

      auto r = std::sqrt(x * x + y * y);
      // amplitude and angle are both default values: 0
      if (std::isnan(r)) continue;

      sm(i, j) = r;

      // std::atan2(0, 0) = 0
      auto a = std::atan2(y, x) / M_PI * 180.; // returns [-180, 180]

      if (((a >= -22.5) && (a < 22.5)) || (a >= 157.5) || (a < -157.5))
        sa(i, j) = value_type(0);
      else if (((a >= 22.5) && (a < 67.5)) || (a >= -157.5) || (a < -112.5))
        sa(i, j) = value_type(45);
      else if (((a >= 67.5) && (a < 112.5)) || (a >= -112.5) || (a < -67.5))
        sa(i, j) = value_type(90);
      else if (((a >= 112.5) && (a < 157.5)) || (a >= -67.5) || (a < -22.5))
        sa(i, j) = value_type(135);
    }
  }

  // non-maximum suppression
  for (size_t i = 1; i < h - 1; ++i)
  {
    for (size_t j = 1; j < w - 1; ++j)
    {
      auto a = sa(i, j);
      auto v = sm(i, j);

      // If amplitude is 0, angle must also be 0.
      if (v == 0) continue;

      // use sa to store the result
      if (a == 0.)
        sa(i, j) = (v >= sm(i-1, j) && v >= sm(i+1, j)) ? v : value_type(0);
      else if (a == 45.)
        sa(i, j) = (v >= sm(i-1, j-1) && v >= sm(i+1, j+1)) ? v : value_type(0);
      else if (a == 90.)
        sa(i, j) = (v >= sm(i, j-1) && v >= sm(i, j+1)) ? v : value_type(0);
      else if (a == 135.)
        sa(i, j) = (v >= sm(i-1, j+1) && v >= sm(i+1, j-1)) ? v : value_type(0);
    }
  }

  // hysteresis thresholding
  using rvalue_type = typename F::value_type;
  for (size_t i = 1; i < h - 1; ++i)
  {
    for (size_t j = 1; j < w - 1; ++j)
    {
      auto v = sa(i, j);
      if (v <= lt)
        dst(i, j) = rvalue_type(0);
      else if (v >= ht)
        dst(i, j) = rvalue_type(1);
      else
        dst(i, j) = static_cast<rvalue_type>((sa(i-1, j) >= ht || sa(i+1, j) >= ht ||
                                              sa(i, j-1) >= ht || sa(i, j+1) >= ht));
    }
  }

}


} // foam

#endif //EXTRA_FOAM_F_CANNY_HPP
