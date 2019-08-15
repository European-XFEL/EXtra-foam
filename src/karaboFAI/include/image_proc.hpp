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

#if defined(FAI_WITH_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#include "tbb/blocked_range3d.h"
#endif

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
 * @param src: image array. shape = (y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename T, template <typename> class C = is_tensor,
  check_container<E, C> = false>
inline void maskPulse(E& src, T lb, T ub)
{
  auto shape = src.shape();

#if defined(FAI_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range2d<int>(0, shape[0], 0, shape[1]),
    [&src, lb, ub] (const tbb::blocked_range2d<int> &block)
    {
      for (int j = block.rows().begin(); j != block.rows().end(); ++j)
      {
        for (int k = block.cols().begin(); k != block.cols().end(); ++k)
        {
#else
      for (size_t j = 0; j < shape[0]; ++j)
      {
        for (size_t k = 0; k < shape[1]; ++k)
        {
#endif
        auto v = src(j, k);
        if (v < lb || v > ub) src(j, k) = T(0);
        }
      }
#if defined(FAI_WITH_TBB)
    }
  );
#endif
}

/**
 * Mask a single image by an image mask inplace.
 *
 * @param src: 2D image array. shape = (y, x)
 * @param mask: 2D image mask. shape = (y, x)
 */
template <typename E, typename M, template <typename> class C = is_tensor,
  check_container<E, C> = false, check_container<M, C> = false>
inline void maskPulse(E& src, const M& mask)
{
  auto shape = src.shape();
  if (shape != mask.shape())
    throw std::invalid_argument("Image and mask have different shapes!");

#if defined(FAI_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range2d<int>(0, shape[0], 0, shape[1]),
    [&src, &mask] (const tbb::blocked_range2d<int> &block)
    {
      for (int j = block.rows().begin(); j != block.rows().end(); ++j)
      {
        for (int k = block.cols().begin(); k != block.cols().end(); ++k)
        {
#else
      for (size_t j = 0; j < shape[0]; ++j)
      {
        for (size_t k = 0; k < shape[1]; ++k)
        {
#endif
          if (mask(j, k)) src(j, k) = 0;
        }
      }
#if defined(FAI_WITH_TBB)
    }
  );
#endif
}

/**
 * Mask images in a train by threshold inplace.
 *
 * @param src: image array. shape = (slices, y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename T, template <typename> class C = is_tensor,
  check_container<E, C> = false>
inline void maskTrain(E& src, T lb, T ub)
{
  auto shape = src.shape();

#if defined(FAI_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range3d<int>(0, shape[0], 0, shape[1], 0, shape[2]),
    [&src, lb, ub] (const tbb::blocked_range3d<int> &block)
    {
      for(int i=block.pages().begin(); i != block.pages().end(); ++i)
      {
        for(int j=block.rows().begin(); j != block.rows().end(); ++j)
        {
          for(int k=block.cols().begin(); k != block.cols().end(); ++k)
          {
#else
      for (size_t i = 0; i < shape[0]; ++i)
      {
        for (size_t j = 0; j < shape[1]; ++j)
        {
          for (size_t k = 0; k < shape[2]; ++k)
          {
#endif
          auto v = src(i, j, k);
          if (v < lb || v > ub) src(i, j, k) = T(0);
          }
        }
      }
#if defined(FAI_WITH_TBB)
    }
  );
#endif
}

/**
 * Mask images in a train by threshold inplace.
 *
 * Pure xtensor implementation.
 *
 * @param src: image array. shape = (slices, y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename T, template <typename> class C = is_tensor,
  check_container<E, C> = false>
inline void xtMaskTrain(E& src, T lb, T ub)
{
  xt::filter(src, src < lb | src > ub) = T(0);
}


/**
 * Mask images in a train by an image mask inplace.
 *
 * @param src: an array of images. shape = (indices, y, x)
 * @param mask: 2D image mask. shape = (y, x)
 */
template <typename E, typename M, template <typename> class C = is_tensor,
  check_container<E, C> = false, check_container<M, C> = false>
inline void maskTrain(E& src, const M& mask)
{
  auto shape = src.shape();
  auto msk_shape = mask.shape();
  if (msk_shape[0] != shape[1] || msk_shape[1] != shape[2])
  {
    throw std::invalid_argument("Image in the train and mask have different shapes!");
  }

#if defined(FAI_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range3d<int>(0, shape[0], 0, shape[1], 0, shape[2]),
    [&src, &mask] (const tbb::blocked_range3d<int> &block)
    {
      for(int i=block.pages().begin(); i != block.pages().end(); ++i)
      {
        for(int j=block.rows().begin(); j != block.rows().end(); ++j)
        {
          for(int k=block.cols().begin(); k != block.cols().end(); ++k)
          {
#else
      for (size_t i = 0; i < shape[0]; ++i)
      {
        for (size_t j = 0; j < shape[1]; ++j)
        {
          for (size_t k = 0; k < shape[2]; ++k)
          {
#endif
          if (mask(j, k)) src(i, j, k) = 0;
          }
        }
      }
#if defined(FAI_WITH_TBB)
    }
  );
#endif
}

template <typename E, template <typename> class C = is_tensor, check_container<E, C> = false>
inline void nanToZeroPulse(E& src)
{
  auto shape = src.shape();

#if defined(FAI_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range2d<int>(0, shape[0], 0, shape[1]),
    [&src] (const tbb::blocked_range2d<int> &block)
    {
      for (int j = block.rows().begin(); j != block.rows().end(); ++j)
      {
        for (int k = block.cols().begin(); k != block.cols().end(); ++k)
        {
#else
      for (size_t j = 0; j < shape[0]; ++j)
      {
        for (size_t k = 0; k < shape[1]; ++k)
        {
#endif
          if (std::isnan(src(j, k))) src(j, k) = 0;
        }
      }
#if defined(FAI_WITH_TBB)
    }
  );
#endif
}

template <typename E, template <typename> class C = is_tensor, check_container<E, C> = false>
inline void nanToZeroTrain(E& src)
{
  auto shape = src.shape();

#if defined(FAI_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range3d<int>(0, shape[0], 0, shape[1], 0, shape[2]),
    [&src] (const tbb::blocked_range3d<int> &block)
    {
      for(int i=block.pages().begin(); i != block.pages().end(); ++i)
      {
        for(int j=block.rows().begin(); j != block.rows().end(); ++j)
        {
          for(int k=block.cols().begin(); k != block.cols().end(); ++k)
          {
#else
      for (size_t i = 0; i < shape[0]; ++i)
      {
        for (size_t j = 0; j < shape[1]; ++j)
        {
          for (size_t k = 0; k < shape[2]; ++k)
          {
#endif
          if (std::isnan(src(i, j, k))) src(i, j, k) = 0;
          }
        }
      }
#if defined(FAI_WITH_TBB)
    }
  );
#endif
}


} // fai

#endif //KARABOFAI_IMAGE_PROC_H
