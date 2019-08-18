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
struct is_pulse : std::false_type {};

template<typename T, xt::layout_type L>
struct is_pulse<xt::xtensor<T, 2, L>> : std::true_type {};

template<typename T>
struct is_train : std::false_type {};

template<typename T, xt::layout_type L>
struct is_train<xt::xtensor<T, 3, L>> : std::true_type {};

template<typename E, template<typename> class C>
using check_container = std::enable_if_t<C<E>::value, bool>;

/**
 * Mask an image by threshold inplace.
 *
 * @param src: image data. shape = (y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename T, template <typename> class C = is_pulse,
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
 * Mask an image by an image mask inplace.
 *
 * @param src: image data. shape = (y, x)
 * @param mask: image mask. shape = (y, x)
 */
template <typename E, typename M, template <typename> class C = is_pulse,
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
 * Mask an array of images by threshold inplace.
 *
 * @param src: image data. shape = (slices, y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename T, template <typename> class C = is_train,
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
 * Mask an array of images by threshold inplace.
 *
 * Pure xtensor implementation.
 *
 * @param src: image data. shape = (slices, y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename T, template <typename> class C = is_train,
  check_container<E, C> = false>
inline void xtMaskTrain(E& src, T lb, T ub)
{
  xt::filter(src, src < lb | src > ub) = T(0);
}


/**
 * Mask an array of images by an image mask inplace.
 *
 * @param src: image data. shape = (indices, y, x)
 * @param mask: image mask. shape = (y, x)
 */
template <typename E, typename M,
  template <typename> class C = is_train, template <typename> class D = is_pulse,
  check_container<E, C> = false, check_container<M, D> = false>
inline void maskTrain(E& src, const M& mask)
{
  auto shape = src.shape();
  auto msk_shape = mask.shape();
  if (msk_shape[0] != shape[1] || msk_shape[1] != shape[2])
  {
    throw std::invalid_argument("Image and mask have different shapes!");
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

/**
 * Inplace converting nan to zero for an image.
 *
 * @param src: image data. shape = (y, x)
 */
template <typename E, template <typename> class C = is_pulse, check_container<E, C> = false>
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

/**
 * Inplace converting nan to zero for an array of images.
 *
 * @param src: image data. shape = (indices, y, x)
 */
template <typename E, template <typename> class C = is_train, check_container<E, C> = false>
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

/**
 * Inplace moving average of an image
 *
 * @param src: moving average of image data. shape = (y, x)
 * @param data: new image data. shape = (y, x)
 * @param count: new moving average count.
 */
template <typename E, template <typename> class C = is_pulse, check_container<E, C> = false>
inline void movingAveragePulse(E& src, const E& data, size_t count)
{
  if (count == 0) throw std::invalid_argument("'count' cannot be zero!");

  auto shape = src.shape();
  if (shape != data.shape())
    throw std::invalid_argument("Inconsistent data shape!");

#if defined(FAI_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range2d<int>(0, shape[0], 0, shape[1]),
    [&src, &data, count] (const tbb::blocked_range2d<int> &block)
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
          src(j, k) += (data(j, k) - src(j, k)) / count;
        }
      }
#if defined(FAI_WITH_TBB)
    }
  );
#endif
}

/**
 * Inplace moving average of an array of images.
 *
 * @param src: moving average of image data. shape = (indices, y, x)
 * @param data: new image data. shape = (indices, y, x)
 * @param count: new moving average count.
 */
template <typename E, template <typename> class C = is_train, check_container<E, C> = false>
inline void movingAverageTrain(E& src, const E& data, size_t count)
{
  if (count == 0) throw std::invalid_argument("'count' cannot be zero!");

  auto shape = src.shape();
  if (shape != data.shape())
    throw std::invalid_argument("Inconsistent data shape!");

#if defined(FAI_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range3d<int>(0, shape[0], 0, shape[1], 0, shape[2]),
    [&src, &data, count] (const tbb::blocked_range3d<int> &block)
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
          src(i, j, k) += (data(i, j, k) - src(i, j, k)) / count;
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
