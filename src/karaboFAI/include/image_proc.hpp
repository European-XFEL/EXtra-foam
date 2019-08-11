/**
 * Offline and online data analysis and visualization tool for azimuthal
 * integration of different data acquired with various detectors at
 * European XFEL.
 *
 * Image processing implemented in C++.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#ifndef KARABOFAI_IMAGE_PROC_H
#define KARABOFAI_IMAGE_PROC_H

#include "xtensor/xtensor.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xindex_view.hpp"


namespace fai {

template<typename T>
struct is_tensor : std::false_type {};

template<typename T, std::size_t N, xt::layout_type L>
struct is_tensor<xt::xtensor<T, N, L>> : std::true_type {};

template<typename T>
struct is_array : std::false_type {};

template<typename T, xt::layout_type L>
struct is_array<xt::xarray<T, L>> : std::true_type {};

template<typename E, template<typename> class C>
using check_container = std::enable_if_t<C<E>::value, bool>;

/**
 * Mask a single image by threshold inplace.
 *
 * @param img: image array. shape = (slices, y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename T, typename E, template <typename> class C = is_tensor,
    check_container<E, C> = false>
inline E maskImage(E& img, T lb, T ub)
{
  xt::filter(img, img < lb | img > ub) = 0;
  return img;
}

/**
 * Mask a single image by an image mask inplace.
 *
 * @param image: 2D image array. shape = (y, x)
 * @param mask: 2D image mask. shape = (y, x)
 */
template <typename E, typename M, template <typename> class C = is_tensor,
    fai::check_container<E, C> = false>
inline E maskImage(E& img, const M& mask)
{
  auto shape = img.shape();
  if (shape != mask.shape())
    throw std::invalid_argument("Image and mask have different shapes!");

  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      if (mask(j, k)) img(j, k) = 0;
    }
  }

  return img;
}

/**
 * Mask images in a train by an image mask inplace.
 *
 * @param img: image array. shape = (slices, y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename T, typename E, template <typename> class C = is_tensor,
    check_container<E, C> = false>
inline E maskTrainImages(E& img, T lb, T ub)
{
  xt::filter(img, img < lb | img > ub) = 0;
  return img;
}

/**
 * Mask images in a train by an image mask inplace.
 *
 * @param image: an array of images. shape = (indices, y, x)
 * @param mask: 2D image mask. shape = (y, x)
 */
template <typename E, typename M, template <typename> class C = is_tensor,
    fai::check_container<E, C> = false>
inline E maskTrainImages(E& img, const M& mask)
{
  auto shape = img.shape();
  auto msk_shape = mask.shape();
  if (msk_shape[0] != shape[1] || msk_shape[1] != shape[2])
  {
    throw std::invalid_argument("Image in the train and mask have different shapes!");
  }

  for (size_t i = 0; i < shape[0]; ++i)
  {
    for (size_t j = 0; j < shape[1]; ++j)
    {
      for (size_t k = 0; k < shape[2]; ++k)
      {
        if (mask(j, k)) img(i, j, k) = 0;
      }
    }
  }

  return img;
}

} // fai

#endif //KARABOFAI_IMAGE_PROC_H
