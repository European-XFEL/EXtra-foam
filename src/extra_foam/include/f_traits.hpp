/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#ifndef EXTRA_FOAM_FOAM_TRAITS_H
#define EXTRA_FOAM_FOAM_TRAITS_H

#include "xtensor/xexpression.hpp"
#include "xtensor/xtensor.hpp"


namespace foam
{

template<typename T>
struct IsImage : std::false_type {};

template<typename T, xt::layout_type L>
struct IsImage<xt::xtensor<T, 2, L>> : std::true_type {};

template<typename T>
struct IsImageMask : std::false_type {};

template<xt::layout_type L>
struct IsImageMask<xt::xtensor<bool, 2, L>> : std::true_type {};

template<typename T>
struct IsImageArray : std::false_type {};

template<typename T, xt::layout_type L>
struct IsImageArray<xt::xtensor<T, 3, L>> : std::true_type {};

template<typename T>
struct IsModulesArray : std::false_type {};

template<typename T, xt::layout_type L>
struct IsModulesArray<xt::xtensor<T, 4, L>> : std::true_type {};

template<typename T>
struct IsModulesVector : std::false_type {};

template<typename T, xt::layout_type L>
struct IsModulesVector<std::vector<xt::xtensor<T, 3, L>>> : std::true_type {};

template<typename E, template<typename> class C>
using EnableIf = std::enable_if_t<C<E>::value, bool>;
}

#endif //EXTRA_FOAM_FOAM_TRAITS_H
