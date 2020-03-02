/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#ifndef EXTRA_FOAM_F_STATISTICS_HPP
#define EXTRA_FOAM_F_STATISTICS_HPP

#include <type_traits>

#include "xtensor/xreducer.hpp"

#if defined(FOAM_WITH_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#include "tbb/blocked_range3d.h"
#endif

#include "f_traits.hpp"


namespace foam
{

template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
inline auto nansum(E&& src)
{
  return xt::nansum(std::forward<E>(src), xt::evaluation_strategy::immediate)[0];
}

template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
inline auto nanmean(E&& src)
{
  return xt::nanmean(std::forward<E>(src), xt::evaluation_strategy::immediate)[0];
}

} // foam


#endif //EXTRA_FOAM_F_STATISTICS_HPP
