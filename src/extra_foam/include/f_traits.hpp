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

#include <type_traits>

#include "xtensor/xexpression.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"


namespace foam
{

template<typename D>
struct IsExpression : std::false_type {};

template<typename D>
struct IsExpression<xt::xexpression<D>> : std::true_type {};

template<typename T>
struct IsArray : std::false_type {};

template<typename T, xt::layout_type L>
struct IsArray<xt::xarray<T, L>> : std::true_type {};

template<typename T>
struct IsVector : std::false_type {};

template<typename T, xt::layout_type L>
struct IsVector<xt::xtensor<T, 1, L>> : std::true_type {};

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
struct IsImageVector : std::false_type {};

template<typename T, xt::layout_type L>
struct IsImageVector<std::vector<xt::xtensor<T, 2, L>>> : std::true_type {};

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

template<typename T, template<typename> typename Test1, template<typename> typename Test2>
using EnableIfEither = std::enable_if_t<std::disjunction_v<Test1<T>, Test2<T>>, bool>;

template<typename E, typename T = void>
using ValueTypeHelper = std::conditional_t<std::is_same_v<T, void>,
                                           typename std::decay_t<E>::value_type,
                                           T>;

template<typename E, typename T, int... axes>
using ReducedTypeHelper = decltype(xt::eval(xt::cast<ValueTypeHelper<E, T>>(xt::sum(std::declval<E>(), {axes...}))));

template<typename E, typename T = void, EnableIf<std::decay_t<E>, IsImage> = false>
using ReducedVectorType = ReducedTypeHelper<E, T, 0>;

template<typename E, typename T = void, EnableIf<std::decay_t<E>, IsImageArray> = false>
using ReducedVectorTypeFromArray = ReducedTypeHelper<E, T, 0, 1>;

template<typename E, typename T = void, EnableIf<std::decay_t<E>, IsImageArray> = false>
using ReducedImageType = ReducedTypeHelper<E, T, 0>;

} // foam

#endif //EXTRA_FOAM_FOAM_TRAITS_H
