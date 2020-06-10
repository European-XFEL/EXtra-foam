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

#include "xtensor/xadapt.hpp"
#include "xtensor/xio.hpp"


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

} //utils
} //foam

#endif //EXTRA_FOAM_UTILITIES_H
