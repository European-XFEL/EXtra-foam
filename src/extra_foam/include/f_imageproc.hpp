/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#ifndef EXTRA_FOAM_IMAGE_PROC_H
#define EXTRA_FOAM_IMAGE_PROC_H

#include <cmath>
#include <thread>
#include <cstring>
#include <type_traits>

#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xindex_view.hpp"

#if defined(FOAM_USE_TBB)
#include "tbb/task_arena.h"
#include "tbb/partitioner.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#endif

#include "f_traits.hpp"
#include "f_utilities.hpp"


namespace foam
{

namespace detail
{

#if defined(FOAM_USE_TBB)
template<typename E>
inline auto nanmeanImageArrayImp(E&& src, const std::vector<size_t>& keep = {})
{
  using value_type = typename std::decay_t<E>::value_type;
  auto shape = src.shape();

  auto mean = ReducedImageType<E>::from_shape({static_cast<std::size_t>(shape[1]),
                                               static_cast<std::size_t>(shape[2])});

  tbb::parallel_for(tbb::blocked_range2d<int>(0, shape[1], 0, shape[2]),
    [&src, &keep, &shape, &mean] (const tbb::blocked_range2d<int> &block)
    {
      for(int j=block.rows().begin(); j != block.rows().end(); ++j)
      {
        for(int k=block.cols().begin(); k != block.cols().end(); ++k)
        {
          std::size_t count = 0;
          value_type sum = 0;
          if (keep.empty())
          {
            for (size_t i=0; i<shape[0]; ++i)
            {
              auto v = src(i, j, k);
              if (! std::isnan(v))
              {
                count += 1;
                sum += v;
              }
            }
          } else
          {
            for (auto it=keep.begin(); it != keep.end(); ++it)
            {
              auto v = src(*it, j, k);
              if (! std::isnan(v))
              {
                count += 1;
                sum += v;
              }
            }
          }

          if (count == 0)
            mean(j, k) = std::numeric_limits<value_type>::quiet_NaN();
          else mean(j, k) = sum / value_type(count);
        }
      }
    }
  );

  return mean;
}
#endif

template<typename E, typename value_type, typename... T>
inline void binPhotonsImp(E const& data, E& out, size_t adu_count, value_type limit, T... indices)
{
  if (std::isnan(data(indices...))) {
    out(indices...) = NAN;
  } else {
    // Bin value
    value_type bin = std::floor((0.5 * adu_count + data(indices...)) / adu_count);
    // Clamp between 0 and infinity
    out(indices...) = std::max(static_cast<value_type>(0), std::min(bin, limit));
  }
}

} // detail

/**
 * Calculate the nanmean of the selected images from an array of images.
 *
 * @param src: image data. shape = (indices, y, x)
 * @param keep: a list of selected indices.
 * @return: the nanmean image. shape = (y, x)
 */
template<typename E, EnableIf<std::decay_t<E>, IsImageArray> = false>
inline auto nanmeanImageArray(E&& src, const std::vector<size_t>& keep)
{
  if (keep.empty()) throw std::invalid_argument("keep cannot be empty!");
#if defined(FOAM_USE_TBB)
  return detail::nanmeanImageArrayImp(std::forward<E>(src), keep);
#else
  using value_type = typename std::decay_t<E>::value_type;
  auto&& sliced(xt::view(std::forward<E>(src), xt::keep(keep), xt::all(), xt::all()));
  return xt::eval(xt::nanmean<value_type>(sliced, 0));
#endif
}

template<typename E, EnableIf<std::decay_t<E>, IsImageArray> = false>
inline auto nanmeanImageArray(E&& src)
{
#if defined(FOAM_USE_TBB)
  return detail::nanmeanImageArrayImp(std::forward<E>(src));
#else
  using value_type = typename std::decay_t<E>::value_type;
  return xt::eval(xt::nanmean<value_type>(std::forward<E>(src), 0));
#endif
}

/**
 * Calculate the nanmean of two images.
 *
 * @param src1: image data. shape = (y, x)
 * @param src2: image data. shape = (y, x)
 * @return: the nanmean image. shape = (y, x)
 */
template<typename E>
inline auto nanmeanImageArray(E&& src1, E&& src2)
{
  using value_type = typename std::decay_t<E>::value_type;
  auto shape = src1.shape();

  utils::checkShape(shape, src2.shape(), "Images have different shapes");

#if defined(FOAM_USE_TBB)
  auto mean = std::decay_t<E>({shape[0], shape[1]});

  tbb::parallel_for(tbb::blocked_range2d<int>(0, shape[0], 0, shape[1]),
    [&src1, &src2, &shape, &mean] (const tbb::blocked_range2d<int> &block)
    {
      for(int j=block.rows().begin(); j != block.rows().end(); ++j)
      {
        for(int k=block.cols().begin(); k != block.cols().end(); ++k)
        {
          auto x = src1(j, k);
          auto y = src2(j, k);

          if (std::isnan(x) and std::isnan(y))
            mean(j, k) = std::numeric_limits<value_type>::quiet_NaN();
          else if (std::isnan(x))
            mean(j, k) = y;
          else if (std::isnan(y))
            mean(j, k) = x;
          else mean(j, k)  = value_type(0.5) * (x + y);
        }
      }
    }
  );

  return mean;
#else
  auto&& stacked = xt::stack(xt::xtuple(std::forward<E>(src1), std::forward<E>(src2)));
  return xt::eval(xt::nanmean<value_type>(stacked, 0));
#endif
}

/**
 * Inplace convert nan using 0 in an image.
 *
 * @param src: image data. shape = (y, x)
 */
template <typename E, EnableIf<E, IsImage> = false>
inline void maskImageDataZero(E& src)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      if (std::isnan(src(j, k))) src(j, k) = value_type(0);
    }
  }
}

/**
 * Maintain an identical API with maskImageDataZero.
 *
 * @param src: image data. shape = (y, x)
 */
template <typename E, EnableIf<E, IsImage> = false>
inline void maskImageDataNan(E& src) {}

/**
 * Get the nan mask of an image.
 *
 * This function is a complementary to maskImageDataNan since the overload
 * maskImageDataNan(E& src, N& out) is ambiguous because of another overload
 * maskImageDataNan(E& src, const M& mask).
 *
 * @param src: image data. shape = (y, x)
 * @param out: output array to place the mask. shape = (y, x)
 */
template <typename E, typename N, EnableIf<E, IsImage> = false, EnableIf<N, IsImageMask> = false>
inline void imageDataNanMask(const E& src, N& out)
{
  auto shape = src.shape();

  utils::checkShape(shape, out.shape(), "Image and output array have different shapes");

  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      if (std::isnan(src(j, k))) out(j, k) = true;
    }
  }
}

/**
 * Inplace mask an image using 0 with threshold mask. Nan pixels in
 * the image are also converted into 0.
 *
 * @param src: image data. shape = (y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename T,
  EnableIf<E, IsImage> = false, std::enable_if_t<std::is_arithmetic<T>::value, bool> = false>
inline void maskImageDataZero(E& src, T lb, T ub)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      auto v = src(j, k);
      if (std::isnan(v) || v < lb || v > ub) src(j, k) = value_type(0);
    }
  }
}

/**
 * Inplace mask an image using 0 with threshold mask. Nan pixels in
 * the image are also converted into 0.
 *
 * @param src: image data. shape = (y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 * @param out: output array to place the overall mask. shape = (y, x)
 */
template <typename E, typename T, typename N,
  EnableIf<E, IsImage> = false, std::enable_if_t<std::is_arithmetic<T>::value, bool> = false,
  EnableIf<N, IsImageMask> = false>
inline void maskImageDataZero(E& src, T lb, T ub, N& out)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, out.shape(), "Image and output array have different shapes");

  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      auto v = src(j, k);
      if (std::isnan(v) || v < lb || v > ub)
      {
        src(j, k) = value_type(0);
        out(j, k) = true;
      }
    }
  }
}

/**
 * Inplace mask an image using nan with threshold mask.
 *
 * @param src: image data. shape = (y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename T,
  EnableIf<E, IsImage> = false, std::enable_if_t<std::is_arithmetic<T>::value, bool> = false>
inline void maskImageDataNan(E& src, T lb, T ub)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      auto v = src(j, k);
      if (std::isnan(v)) continue;
      if (v < lb || v > ub) src(j, k) = nan;
    }
  }
}

/**
 * Inplace mask an image using nan with threshold mask.
 *
 * @param src: image data. shape = (y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 * @param out: output array to place the overall mask. shape = (y, x)
 */
template <typename E, typename T, typename N,
  EnableIf<E, IsImage> = false, std::enable_if_t<std::is_arithmetic<T>::value, bool> = false,
  EnableIf<N, IsImageMask> = false>
inline void maskImageDataNan(E& src, T lb, T ub, N& out)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, out.shape(), "Image and output array have different shapes");

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      auto v = src(j, k);
      if (std::isnan(v)) out(j, k) = true;
      else if (v < lb || v > ub)
      {
        src(j, k) = nan;
        out(j, k) = true;
      }
    }
  }
}

/**
 * Inplace mask an image using 0 with an image mask. Nan pixels in
 * the image are also converted into 0.
 *
 * @param src: image data. shape = (y, x)
 * @param mask: image mask. shape = (y, x)
 */
template <typename E, typename M,
  EnableIf<E, IsImage> = false, EnableIf<M, IsImageMask> = false>
inline void maskImageDataZero(E& src, const M& mask)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes");

  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      if (mask(j, k) || std::isnan(src(j, k))) src(j, k) = value_type(0);
    }
  }
}

/**
 * Inplace mask an image using 0 with an image mask. Nan pixels in
 * the image are also converted into 0.
 *
 * @param src: image data. shape = (y, x)
 * @param mask: image mask. shape = (y, x)
 * @param out: output array to place the overall mask. shape = (y, x)
 */
template <typename E, typename M, typename N,
  EnableIf<E, IsImage> = false, EnableIf<M, IsImageMask> = false, EnableIf<N, IsImageMask> = false>
inline void maskImageDataZero(E& src, const M& mask, M& out)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes");
  utils::checkShape(shape, out.shape(), "Image and output array have different shapes");

  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      if (mask(j, k) || std::isnan(src(j, k)))
      {
        src(j, k) = value_type(0);
        out(j, k) = true;
      }
    }
  }
}

/**
 * Inplace mask an image using nan with an image mask.
 *
 * @param src: image data. shape = (y, x)
 * @param mask: image mask. shape = (y, x)
 */
template <typename E, typename M,
  EnableIf<E, IsImage> = false, EnableIf<M, IsImageMask> = false>
inline void maskImageDataNan(E& src, const M& mask)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes");

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      if (mask(j, k)) src(j, k) = nan;
    }
  }
}

/**
 * Inplace mask an image using nan with an image mask.
 *
 * @param src: image data. shape = (y, x)
 * @param out: output array to place the overall mask. shape = (y, x)
 */
template <typename E, typename M, typename N,
  EnableIf<E, IsImage> = false, EnableIf<M, IsImageMask> = false, EnableIf<N, IsImageMask> = false>
inline void maskImageDataNan(E& src, const M& mask, N& out)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes");
  utils::checkShape(shape, out.shape(), "Image and output array have different shapes");

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      if (std::isnan(src(j, k)))
      {
        out(j, k) = true;
      } else if (mask(j, k))
      {
        src(j, k) = nan;
        out(j, k) = nan;
      }
    }
  }
}

/**
 * Inplace mask an image using 0 with both threshold mask and an image mask.
 * Nan pixels in the image are also converted into 0.
 *
 * @param src: image data. shape = (y, x)
 * @param mask: image mask. shape = (y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename M, typename T,
  EnableIf<E, IsImage> = false, EnableIf<M, IsImageMask> = false,
  std::enable_if_t<std::is_arithmetic<T>::value, bool> = false>
inline void maskImageDataZero(E& src, const M& mask, T lb, T ub)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes");

  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      if (mask(j, k)) src(j, k) = value_type(0);
      else
      {
        auto v = src(j, k);
        if (std::isnan(v) || v < lb || v > ub) src(j, k) = value_type(0);
      }
    }
  }
}

/**
 * Inplace mask an image using 0 with both threshold mask and an image mask.
 * Nan pixels in the image are also converted into 0.
 *
 * @param src: image data. shape = (y, x)
 * @param mask: image mask. shape = (y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 * @param out: output array to place the overall mask. shape = (y, x)
 */
template <typename E, typename M, typename T, typename N,
  EnableIf<E, IsImage> = false, EnableIf<M, IsImageMask> = false, EnableIf<N, IsImageMask> = false>
inline void maskImageDataZero(E& src, const M& mask, T lb, T ub, N& out)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes");
  utils::checkShape(shape, out.shape(), "Image and output array have different shapes");

  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      if (mask(j, k))
      {
        src(j, k) = value_type(0);
        out(j, k) = true;
      } else
      {
        auto v = src(j, k);
        if (std::isnan(v) || v < lb || v > ub)
        {
          src(j, k) = value_type(0);
          out(j, k) = true;
        }
      }
    }
  }
}

/**
 * Inplace mask an image using nan with both threshold mask and an image mask.
 *
 * @param src: image data. shape = (y, x)
 * @param mask: image mask. shape = (y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename M, typename T,
  EnableIf<E, IsImage> = false, EnableIf<M, IsImageMask> = false,
  std::enable_if_t<std::is_arithmetic<T>::value, bool> = false>
inline void maskImageDataNan(E& src, const M& mask, T lb, T ub)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes");

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      if (mask(j, k)) src(j, k) = nan;
      else
      {
        auto v = src(j, k);
        if (std::isnan(v)) continue;
        if (v < lb || v > ub) src(j, k) = nan;
      }
    }
  }
}

/**
 * Inplace mask an image using nan with both threshold mask and an image mask.
 *
 * @param src: image data. shape = (y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 * @param out: output array to place the overall mask. shape = (y, x)
 */
template <typename E, typename M, typename T, typename N,
  EnableIf<E, IsImage> = false, EnableIf<M, IsImageMask> = false, EnableIf<N, IsImageMask> = false>
inline void maskImageDataNan(E& src, const M& mask, T lb, T ub, N& out)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes");
  utils::checkShape(shape, out.shape(), "Image and output array have different shapes");

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      auto v = src(j, k);
      if (mask(j, k))
      {
        src(j, k) = nan;
        out(j, k) = true;
      } else if (std::isnan(v))
      {
        out(j, k) = true;
      } else if (v < lb || v > ub)
      {
        src(j, k) = nan;
        out(j, k) = true;
      }
    }
  }
}

/**
 * Inplace convert nan using 0 in an array of images.
 *
 * @param src: image data. shape = (indices, y, x)
 */
template <typename E, EnableIf<E, IsImageArray> = false>
inline void maskImageDataZero(E& src)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(0, shape[0]),
    [&src, &shape] (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
        for(size_t j=0; j != shape[1]; ++j)
        {
          for(size_t k=0; k != shape[2]; ++k)
          {
            if (std::isnan(src(i, j, k))) src(i, j, k) = value_type(0);
          }
        }
      }
    }
  );
#else
  xt::filter(src, xt::isnan(src)) = value_type(0);
#endif
}

/**
 * Maintain an identical API with maskImageDataZero.
 *
 * @param src: image data. shape = (indices, y, x)
 */
template <typename E, EnableIf<E, IsImageArray> = false>
inline void maskImageDataNan(E& src) {}

/**
 * Inplace mask an array of images using 0 with threshold mask. Nan pixels in
 * those images are also converted into 0.
 *
 * @param src: image data. shape = (slices, y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename T,
  EnableIf<E, IsImageArray> = false, std::enable_if_t<std::is_arithmetic<T>::value, bool> = false>
inline void maskImageDataZero(E& src, T lb, T ub)
{
  using value_type = typename E::value_type;
#if defined(FOAM_USE_TBB)
  auto shape = src.shape();

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
  tbb::parallel_for(tbb::blocked_range<int>(0, shape[0]),
    [&src, lb, ub, nan, &shape] (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
        for(size_t j=0; j != shape[1]; ++j)
        {
          for(size_t k=0; k != shape[2]; ++k)
          {
            auto v = src(i, j, k);
            if (std::isnan(v) || v < lb || v > ub) src(i, j, k) = value_type(0);
          }
        }
      }
    }
  );
#else
  xt::filter(src, xt::isnan(src) | src < lb | src > ub) = value_type(0);
#endif
}

/**
 * Inplace mask an array of images using nan with threshold mask.
 *
 * @param src: image data. shape = (slices, y, x)
 * @param lb: lower threshold
 * @param ub: upper threshold
 */
template <typename E, typename T,
  EnableIf<E, IsImageArray> = false, std::enable_if_t<std::is_arithmetic<T>::value, bool> = false>
inline void maskImageDataNan(E& src, T lb, T ub)
{
  using value_type = typename E::value_type;

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
#if defined(FOAM_USE_TBB)
  auto shape = src.shape();

  tbb::parallel_for(tbb::blocked_range<int>(0, shape[0]),
    [&src, lb, ub, nan, &shape] (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
        for(size_t j=0; j != shape[1]; ++j)
        {
          for(size_t k=0; k != shape[2]; ++k)
          {
            auto v = src(i, j, k);
            if (std::isnan(v)) continue;
            if (v < lb || v > ub) src(i, j, k) = nan;
          }
        }
      }
    }
  );
#else
  xt::filter(src, src < lb | src > ub) = nan;
#endif
}

/**
 * Inplace mask an array of images using 0 with an image mask. Nan pixels in
 * those images are also converted into 0.
 *
 * @param src: image data. shape = (indices, y, x)
 * @param mask: image mask. shape = (y, x)
 */
template <typename E, typename M,
  EnableIf<E, IsImageArray> = false, EnableIf<M, IsImageMask> = false>
inline void maskImageDataZero(E& src, const M& mask)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes", 1);

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(0, shape[0]),
    [&src, &mask, nan, &shape] (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
#else
      for (size_t i = 0; i < shape[0]; ++i)
      {
#endif
        for (size_t j = 0; j < shape[1]; ++j)
        {
          for (size_t k = 0; k < shape[2]; ++k)
          {
            if (mask(j, k) || std::isnan(src(i, j, k))) src(i, j, k) = value_type(0);
          }
        }
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif
}

/**
 * Inplace mask an array of images using nan with an image mask.
 *
 * @param src: image data. shape = (indices, y, x)
 * @param mask: image mask. shape = (y, x)
 */
template <typename E, typename M,
  EnableIf<E, IsImageArray> = false, EnableIf<M, IsImageMask> = false>
inline void maskImageDataNan(E& src, const M& mask)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes", 1);

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(0, shape[0]),
    [&src, &mask, nan, &shape] (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
#else
      for (size_t i = 0; i < shape[0]; ++i)
      {
#endif
        for (size_t j = 0; j < shape[1]; ++j)
        {
          for (size_t k = 0; k < shape[2]; ++k)
          {
            if (mask(j, k)) src(i, j, k) = nan;
          }
        }
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif
}

/**
 * Inplace mask an array of images using 0 with both threshold mask and an image mask.
 * Nan pixels in those images are also converted into 0.
 *
 * @param src: image data. shape = (indices, y, x)
 * @param mask: image mask. shape = (y, x)
 */
template <typename E, typename M, typename T,
  EnableIf<E, IsImageArray> = false, EnableIf<M, IsImageMask> = false>
inline void maskImageDataZero(E& src, const M& mask, T lb, T ub)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes", 1);

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(0, shape[0]),
    [&src, &mask, lb, ub, nan, &shape] (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
#else
      for (size_t i = 0; i < shape[0]; ++i)
      {
#endif
        for (size_t j = 0; j < shape[1]; ++j)
        {
          for (size_t k = 0; k < shape[2]; ++k)
          {
            if (mask(j, k))
            {
              src(i, j, k) = value_type(0);
            } else
            {
              auto v = src(i, j, k);
              if (std::isnan(v) || v < lb || v > ub) src(i, j, k) = value_type(0);
            }
          }
        }
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif
}

/**
 * Inplace mask an array of images using nan with both threshold mask and an image mask.
 *
 * @param src: image data. shape = (indices, y, x)
 * @param mask: image mask. shape = (y, x)
 */
template <typename E, typename M, typename T,
  EnableIf<E, IsImageArray> = false, EnableIf<M, IsImageMask> = false>
inline void maskImageDataNan(E& src, const M& mask, T lb, T ub)
{
  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, mask.shape(), "Image and mask have different shapes", 1);

  auto nan = std::numeric_limits<value_type>::quiet_NaN();
#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(0, shape[0]),
    [&src, &mask, lb, ub, nan, &shape] (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
#else
      for (size_t i = 0; i < shape[0]; ++i)
      {
#endif
        for (size_t j = 0; j < shape[1]; ++j)
        {
          for (size_t k = 0; k < shape[2]; ++k)
          {
            if (mask(j, k))
            {
              src(i, j, k) = nan;
            } else
            {
              auto v = src(i, j, k);
              if (std::isnan(v)) continue;
              if (v < lb || v > ub) src(i, j, k) = nan;
            }
          }
        }
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif
}

template<typename E, EnableIfEither<E, IsImage, IsImageArray> = false>
inline void binPhotons(const E& data, size_t adu_count, E& out)
{
  using value_type = typename std::decay_t<E>::value_type;
  value_type limit = std::numeric_limits<value_type>::infinity();
  auto shape = data.shape();

  if constexpr (data.rank == 2) {
    utils::applyFunctor2d(shape, [&] (size_t i, size_t j) {
      detail::binPhotonsImp(data, out, adu_count, limit, i, j);
    });
  } else if (data.rank == 3) {
    utils::applyFunctor3d(shape, [&] (size_t i, size_t j, size_t k) {
      detail::binPhotonsImp(data, out, adu_count, limit, i, j, k);
    });
  }
}

  template<typename A>
      inline auto generateAssemblyLUT(size_t data_size, const A& assembled)
  {
      auto lut = xt::eval(xt::zeros<size_t>({data_size}));

      for (size_t i = 0; i < assembled.size(); ++i) {
          if (!std::isnan(assembled.flat(i))) {
              lut(static_cast<size_t>(assembled.flat(i))) = i;
          }
      }

      return lut;
  }

  template<typename A>
      inline auto generateAssemblyLUT2(size_t data_size, const A& assembled)
  {
      auto lut = xt::eval(xt::zeros<uint8_t>({data_size * 3}));

      for (uint32_t i = 0; i < assembled.size(); ++i) {
          if (!std::isnan(assembled.flat(i))) {
              size_t idx_start = static_cast<size_t>(assembled.flat(i)) * 3;

              // Pack the first 24 bytes of the 32-bit int into the LUT
              std::memcpy(&(lut(idx_start)), &i, 3);
          }
      }

      return lut;
  }

  /**
   * Attempt 1: Linear LUT the size of the total input size.
   *
   * Too slow, LUT takes precious space on the CPU cache.
   */
  template<typename M, typename A, typename L>
      inline void assembleDetectorData(const M& modules, A& assembled, const L& lut, int j)
  {
      if (lut.size() != modules.size()) {
          throw std::invalid_argument("LUT is not the same size as the module array");
      } else if (!modules.is_contiguous() || !assembled.is_contiguous() || !lut.is_contiguous() ||
                 modules.layout() == xt::layout_type::dynamic ||
                 !(modules.layout() == assembled.layout() && assembled.layout() == lut.layout())) {
          throw std::invalid_argument("Non-contiguous inputs");
      }

      auto modules_array = modules.data();
      auto assembled_array = assembled.data();

      tbb::task_arena arena{j};
      arena.initialize();
      arena.execute([&] {
          tbb::parallel_for(tbb::blocked_range<size_t>(0, modules.size()),
                            [&] (const tbb::blocked_range<size_t> &block) {
                                for (size_t i = block.begin(); i != block.end(); ++i) {
                                    assembled_array[lut(i)] = modules_array[i];
                                }
                            });
      });
  }

  /**
   * Attempt 2: Use tile corner coordinates from extra-geom.
   *
   * This is unparallelized, but offers better performance than the first
   * attempt. Need to figure out in which order to copy the module data though,
   * and that's complicated.
   */
  template<typename M, typename A, typename L>
      inline void assembleDetectorData2(const M& modules, A& assembled,
                                        const L& lut,
                                        const L& lut_rows, const L& lut_cols,
                                        size_t tile_rows, size_t tile_cols)
  {
      auto modules_buffer = modules.data();
      auto assembled_buffer = assembled.data();

      auto dest_strides = assembled.strides();
      size_t row_len = assembled.shape()[2];
      size_t tiles_per_module = lut_rows.shape()[1];
      size_t tile_row_len = static_cast<size_t>(modules.shape()[2] / tiles_per_module);

      auto kernel = [&] (size_t pulse, size_t module) {
          for (size_t tile = 0; tile < tiles_per_module; ++tile) {
              size_t start = pulse * dest_strides[0] + lut(module, tile);
              for (size_t row = 0; row < tile_rows; ++row) {
                  for (size_t col = 0; col < tile_cols; ++col) {
                      assembled_buffer[start + row * row_len + col] = modules(pulse, module, row, (tile * tile_cols) + col);
                  }
              }
          }
      };

      size_t n_pulses = modules.shape()[0];
      size_t n_modules = modules.shape()[1];

      for (size_t pulse = 0; pulse < n_pulses; ++pulse) {
          for (size_t module = 0; module < n_modules; ++module) {
              kernel(pulse, module);
          }
      }

      // tbb::parallel_for(tbb::blocked_range2d<size_t>(0, n_pulses, 0, n_pulses),
      //                   [&] (const tbb::blocked_range2d<size_t>& block) {
      //                       for (size_t pulse = block.rows().begin(); pulse != block.rows().end(); ++pulse) {
      //                           for (size_t module = block.cols().begin(); module != block.cols().end(); ++module) {
      //                               kernel(pulse, module);
      //                           }
      //                       }
      //                   });
  }

  /**
   * Attempt 3: Copy data into a column-major layout array to take advantage of
   * cache locality.
   *
   * Can't use a (pulses, y, x) shaped output array, because then the elements
   * for each pixel in a pulse will not be contiguous. If we're doing it this
   * way, then we'll need a list of n (x, y) arrays.
   */
  template<typename M, typename A, typename L>
      inline void assembleDetectorData3(const M& src, A& dest,
                                        const L& lut,
                                        const L& lut_rows, const L& lut_cols,
                                        size_t tile_rows, size_t tile_cols)
  {
      if (!dest.is_contiguous() || dest.layout() != xt::layout_type::column_major) {
          throw std::invalid_argument("Destination array does not have column-major layout");
      }

      auto src_buffer = src.data();
      auto dest_buffer = dest.data();

      auto dest_strides = dest.strides();
      size_t col_len = dest.shape()[1];
      size_t tiles_per_module = lut_rows.shape()[1];

      for (size_t i = 0; i < dest.size(); ++i) {
          dest_buffer[i] = static_cast<float>(i);
      }
      return;

      auto kernel = [&] (size_t pulse, size_t module) {
          for (size_t tile = 0; tile < tiles_per_module; ++tile) {
              size_t start = pulse * dest_strides[0] + lut(module, tile);

              for (size_t col = 0; col < tile_cols; ++col) {
                  for (size_t row = 0; row < tile_rows; ++row) {
                      dest_buffer[start + col * col_len + row] = src(pulse, module, (tile * tile_cols) + col, row);
                  }
              }
          }
      };

      size_t n_pulses = src.shape()[0];
      size_t n_modules = src.shape()[1];
      for (size_t pulse = 0; pulse < n_pulses; ++pulse) {
          for (size_t module = 0; module < n_modules; ++module) {
              kernel(pulse, module);
          }
      }
  }

  /**
   * Attempt 4: Linear LUT for a single pulse, reused for each pulse.
   *
   * Very close (within ~5-7%) to matching the current implementations'
   * performance. Limitation is the amount of space used in the cache by the
   * LUT.
   */
  template<typename M, typename A, typename L>
      inline void assembleDetectorData4(const M& modules, A& assembled, const L& lut, int j)
  {
      if (!modules.is_contiguous() || !assembled.is_contiguous() || !lut.is_contiguous() ||
          modules.layout() == xt::layout_type::dynamic ||
          !(modules.layout() == assembled.layout() && assembled.layout() == lut.layout())) {
          throw std::invalid_argument("Non-contiguous inputs");
      }

      static tbb::affinity_partitioner ap{ };

      auto modules_array = modules.data();
      auto assembled_array = assembled.data();
      auto lut_array = lut.data();

      size_t src_pulse_stride = modules.strides()[0];
      size_t dest_pulse_stride = assembled.strides()[0];

      tbb::task_arena arena{j};
      arena.initialize();
      arena.execute([&] {
          tbb::parallel_for(tbb::blocked_range<size_t>(0, modules.size()),
                            [&] (const tbb::blocked_range<size_t> &block) {
                                for (size_t i = block.begin(); i != block.end(); ++i) {
                                    size_t pulse = i >> 20;
                                    size_t src_offset = pulse << 20;
                                    size_t dest_offset = dest_pulse_stride * pulse;
                                    assembled_array[dest_offset + lut_array[i - src_offset]] = modules_array[i];
                                }
                            },
                            ap);
      });
  }

  /**
   * Attempt 5: Linear LUT for a single pulse, but packed into 24-bit ints for
   * minimal effect on the cache.
   *
   * Good performance, very close to that of the current assembler. Consistenly
   * faster than attempt 4 by a couple percentage points, and more stable.
   */
  template<typename M, typename A, typename L>
      void assembleDetectorData5(const M& modules, A& assembled, const L& lut, int j)
  {
      if (!modules.is_contiguous() || !assembled.is_contiguous() || !lut.is_contiguous() ||
          modules.layout() == xt::layout_type::dynamic ||
          !(modules.layout() == assembled.layout() && assembled.layout() == lut.layout())) {
          throw std::invalid_argument("Non-contiguous inputs");
      }

      auto modules_array = modules.data();
      auto assembled_array = assembled.data();
      auto lut_array = lut.data();

      size_t src_pulse_stride = modules.strides()[0];
      size_t dest_pulse_stride = assembled.strides()[0];

      tbb::task_arena arena{j};
      arena.initialize();
      arena.execute([&] {
          tbb::parallel_for(tbb::blocked_range<size_t>(0, modules.size()),
                            [&] (const tbb::blocked_range<size_t> &block) {
                                for (size_t i = block.begin(); i != block.end(); ++i) {
                                    size_t pulse = i >> 20;
                                    size_t src_offset = pulse << 20;
                                    size_t dest_offset = dest_pulse_stride * pulse;

                                    // Unpack the LUT value
                                    uint32_t lut_idx_start = (i - src_offset) * 3;
                                    uint32_t lut_value = 0;
                                    std::memcpy(&lut_value, &lut_array[lut_idx_start], 4);
                                    lut_value &= 0x00ffffff;

                                    // Copy data
                                    assembled_array[dest_offset + lut_value] = modules_array[i];
                                }
                            });
      });
  }

  /**
   * Attempt 6: Same as attempt 5, but with slightly more advanced code to only
   * compute the LUT index when necessary.
   *
   * This consistently (but barely) outperforms attempt 5 by a margin of a
   * couple percent. The performance also seems a bit more stable compared
   * to attempt 5.
   */
  template<typename M, typename A, typename L>
      void assembleDetectorData6(const M& modules, A& assembled, const L& lut)
  {
      if (!modules.is_contiguous() || !assembled.is_contiguous() || !lut.is_contiguous() ||
          modules.layout() == xt::layout_type::dynamic ||
          !(modules.layout() == assembled.layout() && assembled.layout() == lut.layout())) {
          throw std::invalid_argument("Non-contiguous inputs");
      }

      auto modules_array = modules.data();
      auto assembled_array = assembled.data();
      auto lut_array = lut.data();

      size_t src_pulse_stride = modules.strides()[0];
      size_t dest_pulse_stride = assembled.strides()[0];

      // This is a helper function to calculate the right index within the LUT
      // for the current pixel.
      auto compute_lut_idx = [dest_pulse_stride] (const size_t& module_idx, size_t& src_offset,
                                                  size_t& dest_offset, uint32_t& lut_idx) {
          // This is a trick that takes advantage of the fact that the 1M
          // detectors (AGIPD/LPD) have exactly 2^20 pixels. Since in memory the
          // module data for each pulse is in order after each other, we can get
          // the pulse number by dividing the current location in the 1D array
          // by the number of pixels per pulse. And since this is a multiple of
          // two we can cheat and do a bitshift instead, which is faster than
          // division.
          size_t pulse = module_idx >> 20;

          // Multiply by the pulse data length to get the index in the source
          // data of the beginning of the current pulse. Again, since the number
          // of pixels in a pulse is a power of 2 we can bitshift.
          src_offset = pulse << 20;

          // In the destination, the offset to the beginning of the current
          // pulse is just the stride length of a pulse multiplied by the pulse
          // number. The stride length may not be a power of 2 so we can't
          // bitshift here.
          dest_offset = dest_pulse_stride * pulse;

          // Finally, calculate the index in the LUT for the lookup value for
          // the current pixel. This is the offset within the current pulse,
          // multiplied by 3 (because the LUT values take up 3 bytes in the
          // array).
          lut_idx = (module_idx - src_offset) * 3;
      };

      // Using an affinity_partitioner significantly (~10-30%) improves
      // performance on Maxwell for trains with less than ~150 pulses.
      static tbb::affinity_partitioner ap{ };
      // On Maxwell this gives the best performance, though on more 'consumer'
      // machines performance slightly improves (~10%) with half this number.
      tbb::task_arena arena{std::thread::hardware_concurrency()};
      arena.initialize();
      arena.execute([&] {
          tbb::parallel_for(tbb::blocked_range<size_t>(0, modules.size()),
                            [&] (const tbb::blocked_range<size_t> &block) {
                                // Check if this block is within a single pulse
                                size_t start = block.begin();
                                size_t start_pulse = start >> 20;
                                size_t end = block.end();
                                size_t end_pulse = end >> 20;
                                bool intra_pulse = start_pulse == end_pulse;

                                size_t src_offset = 0;
                                size_t dest_offset = 0;
                                uint32_t lut_idx_start = 0;
                                compute_lut_idx(start, src_offset, dest_offset, lut_idx_start);

                                for (size_t i = block.begin(); i != block.end(); ++i) {
                                    // Compute the right LUT index. If the block
                                    // we're processing is within a pulse then
                                    // we can just increment the LUT index.
                                    if (intra_pulse) {
                                        lut_idx_start += 3;
                                    } else {
                                        // Otherwise we do the full calculation
                                        compute_lut_idx(i, src_offset, dest_offset, lut_idx_start);
                                    }

                                    // Unpack the LUT value. Each value is a
                                    // 24-bit int so technically we only need to
                                    // copy 3 bytes, but copying 4 bytes is
                                    // about 5x faster because then the compiler
                                    // can optimize it down to a single assembly
                                    // instruction.
                                    uint32_t lut_value = 0;
                                    std::memcpy(&lut_value, &lut_array[lut_idx_start], 4);
                                    // Now we've copied 4 bytes into this int,
                                    // but we only want the first 3 so we mask
                                    // the high bits. This gives us the final
                                    // value.
                                    lut_value &= 0x00ffffff;

                                    // Copy the pixel into the right position
                                    assembled_array[dest_offset + lut_value] = modules_array[i];
                                }
                            },
                            ap);
      });
  }

/**
 * Inplace apply moving average for an image
 *
 * @param src: moving average of image data. shape = (y, x)
 * @param data: new image data. shape = (y, x)
 * @param count: new moving average count.
 */
template <typename E, EnableIf<E, IsImage> = false>
inline void movingAvgImageData(E& src, const E& data, size_t count)
{
  if (count == 0) throw std::invalid_argument("'count' cannot be zero!");

  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, data.shape(), "Inconsistent data shapes");

  for (size_t j = 0; j < shape[0]; ++j)
  {
    for (size_t k = 0; k < shape[1]; ++k)
    {
      src(j, k) += (data(j, k) - src(j, k)) / value_type(count);
    }
  }
}

/**
 * Inplace apply moving average for an array of images.
 *
 * @param src: moving average of image data. shape = (indices, y, x)
 * @param data: new image data. shape = (indices, y, x)
 * @param count: new moving average count.
 */
template <typename E, EnableIf<E, IsImageArray> = false>
inline void movingAvgImageData(E& src, const E& data, size_t count)
{
  if (count == 0) throw std::invalid_argument("'count' cannot be zero!");

  using value_type = typename E::value_type;
  auto shape = src.shape();

  utils::checkShape(shape, data.shape(), "Inconsistent data shapes");

#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(0, shape[0]),
    [&src, &data, count, &shape] (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
#else
      for (size_t i = 0; i < shape[0]; ++i)
      {
#endif
        for (size_t j = 0; j < shape[1]; ++j)
        {
          for (size_t k = 0; k < shape[2]; ++k)
          {
            src(i, j, k) += (data(i, j, k) - src(i, j, k)) / value_type(count);
          }
        }
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif
}

class OffsetPolicy
{
public:
  template<typename E1, typename E2>
  static void correct(E1& src, const E2& offset)
  {
    // TODO:: simplify after xtensor-python bug fixing
    auto shape = src.shape();
    for (size_t j = 0; j < shape[0]; ++j)
    {
      for (size_t k = 0; k < shape[1]; ++k)
      {
        src(j, k) -= offset(j, k);
      }
    }
  }
};

/**
 * DSSC raw data has dark values around 40. However, pixels with values
 * of 256 are sometimes saved as 0.
 */
class DsscOffsetPolicy
{
public:
  template<typename E1, typename E2>
  static void correct(E1& src, const E2& offset)
  {
    // TODO:: simplify after xtensor-python bug fixing
    auto shape = src.shape();
    using value_type = typename E1::value_type;
    for (size_t j = 0; j < shape[0]; ++j)
    {
      for (size_t k = 0; k < shape[1]; ++k)
      {
        src(j, k) = (src(j, k) ? src(j, k) : value_type(256)) - offset(j, k);
      }
    }
  }
};

class GainPolicy
{
public:
  template<typename E1, typename E2>
  static void correct(E1& src, const E2& gain)
  {
    // TODO:: simplify after xtensor-python bug fixing
    auto shape = src.shape();
    for (size_t j = 0; j < shape[0]; ++j)
    {
      for (size_t k = 0; k < shape[1]; ++k)
      {
        src(j, k) *= gain(j, k);
      }
    }
  }
};

class GainOffsetPolicy
{
public:
  template<typename E1, typename E2, typename E3>
  static void correct(E1& src, const E2& gain, const E3& offset)
  {
    // TODO:: simplify after xtensor-python bug fixing
    auto shape = src.shape();
    for (size_t j = 0; j < shape[0]; ++j)
    {
      for (size_t k = 0; k < shape[1]; ++k)
      {
        src(j, k) = gain(j, k) * ( src(j, k) - offset(j, k) );
      }
    }
  }
};

namespace detail
{

template <typename Policy, typename E>
inline void correctImageDataImp(E& src, const E& constants)
{
  auto shape = src.shape();
#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(0, shape[0]),
    [&src, &constants, &shape] (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
#else
    for (size_t i = 0; i < shape[0]; ++i)
      {
#endif
        auto&& src_view = xt::view(src, i, xt::all(), xt::all());
        Policy::correct(src_view, xt::view(constants, i, xt::all(), xt::all()));
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif
}

} //detail

/**
 * Inplace apply either gain or offset correct for an array of images.
 *
 * @tparam Policy: correction policy (OffsetPolicy or GainPolicy)
 *
 * @param src: image data. shape = (indices, y, x)
 * @param constants: correction constants, which has the same shape as src.
 */
template <typename Policy, typename E, EnableIf<E, IsImageArray> = false>
inline void correctImageData(E& src, const E& constants)
{
  utils::checkShape(src.shape(), constants.shape(), "data and constants have different shapes");

  detail::correctImageDataImp<Policy>(src, constants);
}

/**
* Inplace apply either gain or offset correct for an image.
*
* @tparam Policy: correction policy (OffsetPolicy or GainPolicy)
*
* @param src: image data. shape = (y, x)
* @param constants: correction constants, which has the same shape as src.
*/
template <typename Policy, typename E, EnableIf<E, IsImage> = false>
inline void correctImageData(E& src, const E& constants)
{
  utils::checkShape(src.shape(), constants.shape(), "data and constants have different shapes");
  Policy::correct(src, constants);
}

/**
* Inplace apply both gain and offset correct for an array of images.
*
* @param src: image data. shape = (indices, y, x)
* @param gain: gain correction constants, which has the same shape as src.
* @param offset: offset correction constants, which has the same shape as src.
*/
template <typename Policy, typename E, EnableIf<E, IsImageArray> = false>
inline void correctImageData(E& src, const E& gain, const E& offset)
{
  auto shape = src.shape();

  utils::checkShape(shape, gain.shape(), "data and gain constants have different shapes");
  utils::checkShape(shape, offset.shape(), "data and offset constants have different shapes");

#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(0, shape[0]),
    [&src, &gain, &offset, &shape] (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
#else
      for (size_t i = 0; i < shape[0]; ++i)
      {
#endif
        auto&& src_view = xt::view(src, i, xt::all(), xt::all());
        Policy::correct(src_view,
                        xt::view(gain, i, xt::all(), xt::all()),
                        xt::view(offset, i, xt::all(), xt::all()));
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif
}

/**
* Inplace apply both gain and offset correct for an image.
*
* @param src: image data. shape = (y, x)
* @param gain: gain correction constants, which has the same shape as src.
* @param offset: offset correction constants, which has the same shape as src.
*/
template <typename Policy, typename E, EnableIf<E, IsImage> = false>
inline void correctImageData(E& src, const E& gain, const E& offset)
{
  auto shape = src.shape();

  utils::checkShape(shape, gain.shape(), "data and gain constants have different shapes");
  utils::checkShape(shape, offset.shape(), "data and offset constants have different shapes");

  Policy::correct(src, gain, offset);
}

/**
 * Inplace apply interleaved intra-dark correction for an array of images.
 *
 * @param src: image data. shape = (indices, y, x)
 */
template <typename E, EnableIf<E, IsImageArray> = false>
inline void correctImageData(E& src)
{
  utils::checkEven(src.shape()[0], "Number of images must be an even number");

  auto&& src_view = xt::view(src, xt::range(0, xt::placeholders::_, 2), xt::all(), xt::all());
  detail::correctImageDataImp<OffsetPolicy>(
    src_view, xt::view(src, xt::range(1, xt::placeholders::_, 2), xt::all(), xt::all()));
}

} // foam

#endif //EXTRA_FOAM_IMAGE_PROC_H
