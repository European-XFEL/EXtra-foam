/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#ifndef EXTRA_FOAM_UTILITIES_H
#define EXTRA_FOAM_UTILITIES_H

#include <string>
#include <algorithm>
#include <functional>

#include "xtensor/xadapt.hpp"
#include "xtensor/xio.hpp"

#if defined(FOAM_USE_TBB)
#include "tbb/blocked_range2d.h"
#include "tbb/blocked_range3d.h"
#endif


namespace foam
{
namespace utils
{

/**
 * Compare two shape containers.
 *
 * @param shape1: container for shape 1.
 * @param shape2: container for shape 2.
 * @param header: header for the error message if the two shapes are different.
 * @param s0: starting position of the first container.
 * @param s1: starting position of the second container.
 */
template<typename S1, typename S2>
inline void checkShape(S1&& shape1, S2&& shape2, std::string&& header, size_t s0=0, size_t s1=0)
{
  if (not std::equal(shape1.begin() + s0, shape1.end(), shape2.begin() + s1))
  {
    std::ostringstream ss;
    ss << header << ": " << xt::adapt(shape1) << " and " << xt::adapt(shape2);
    throw std::invalid_argument(ss.str());
  }
}

template<typename T>
inline void checkEven(T a, std::string&& header)
{
  if (a % 2 != 0)
  {
    std::ostringstream ss;
    ss << header << ": " << a;
    throw std::invalid_argument(ss.str());
  }
}

template<typename shape_t>
inline void applyFunctor2d(const std::array<shape_t, 2>& shape, std::function<void(size_t, size_t)> functor)
{
#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range2d<size_t>(0, shape[0], 0, shape[1]),
    [&functor] (const tbb::blocked_range2d<size_t> &block) {
      for (size_t i = block.rows().begin(); i != block.rows().end(); ++i) {
        for (size_t j = block.cols().begin(); j != block.cols().end(); ++j) {
#else
      for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
#endif
          functor(i, j);
        }
      }
#if defined(FOAM_USE_TBB)
    });
#endif
}

template<typename shape_t>
inline void applyFunctor3d(const std::array<shape_t, 3>& shape, std::function<void(size_t, size_t, size_t)> functor)
{
#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range3d<size_t>(0, shape[0], 0, shape[1], 0, shape[2]),
    [&functor] (const tbb::blocked_range3d<size_t> &block) {
      for (size_t i = block.pages().begin(); i != block.pages().end(); ++i) {
        for (size_t j = block.rows().begin(); j != block.rows().end(); ++j) {
          for (size_t k = block.cols().begin(); k != block.cols().end(); ++k) {
#else
      for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
          for (size_t k = 0; k < shape[2]; ++k) {
#endif
            functor(i, j, k);
          }
        }
      }
#if defined(FOAM_USE_TBB)
    });
#endif
}

} //utils
} //foam

#endif //EXTRA_FOAM_UTILITIES_H
