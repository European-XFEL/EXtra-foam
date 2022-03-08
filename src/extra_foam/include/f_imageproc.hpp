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
#include <type_traits>

#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xindex_view.hpp"

#if defined(FOAM_USE_TBB)
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

template<typename E, typename L, EnableIf<E, IsImage> = false>
inline void dropletize(E& image, const L& labelled_image, E& photon_data, size_t n_labels, float adu_count)
{
    auto count_photons = [&] (auto& indices) {
        float label_sum{ 0 };

        for (auto& i : indices) {
            label_sum += image.flat(i);
        }

        return static_cast<int>(std::round(label_sum / adu_count));
    };

    auto idxs_weighted_mean = [&] (auto& indices) {
        std::array<float, 2> weighted_sum{ 0, 0 };
        float weights_sum{ 0 };

        for (auto& flat_idx : indices) {
            auto array_idx{ xt::unravel_index(flat_idx, image.shape()) };

            weighted_sum[0] += array_idx[0] * image[array_idx];
            weighted_sum[1] += array_idx[1] * image[array_idx];
            weights_sum += image[array_idx];
        }

        return std::array<size_t, 2>{ static_cast<size_t>(std::round(weighted_sum[0] / weights_sum)),
                                   static_cast<size_t>(std::round(weighted_sum[1] / weights_sum)) };
    };

    // This vector of photon counts has one more entry than necessary so
    // we can use base-1 indexing for convenience.
    std::vector<float> photon_counts(n_labels + 1, 0);

    // Do an initial photon count of all labels
    for (int i = 0; i < image.size(); ++i) {
        int label = labelled_image.flat(i);
        photon_counts[label] += image.flat(i);
    }
    for (auto& count : photon_counts) {
        count = std::round(count / adu_count);
    }

    // Iterate over each droplet in parallel.
    // Known bug: the photon counts may be very slightly (e.g. 1 or 2) off
    // when running in parallel. This is likely coming from the code where
    // we check the neighbouring pixels, which does not consider whether
    // the neighbours belong to a different label.
    tbb::parallel_for(size_t(1), n_labels + 1,
                      [&] (int label) {
                          auto array_idxs{ xt::argwhere(xt::equal(labelled_image, label)) };
                          auto label_indices{ xt::ravel_indices<xt::ravel_vector_tag>(array_idxs, labelled_image.shape()) };
                          std::sort(label_indices.begin(), label_indices.end());

                          int photons{ static_cast<int>(photon_counts[label]) };

                          // If the droplet has more than 1 photon, assign 1 to the pixels
                          // with high enough ADUs. Once a photon is assigned, remove the
                          // ADU from the corresponding pixel.
                          if (photons > 1) {
                              for (int pixel_idx : label_indices) {
                                  float& pixel{ image.flat(pixel_idx) };
                                  float photons_in_pixel = std::floor(pixel / adu_count);
                                  pixel -= photons_in_pixel * adu_count;
                                  photon_data.flat(pixel_idx) += photons_in_pixel;
                              }

                              // Re-count the number of photons
                              photons = count_photons(label_indices);
                          }

                          if (photons == 0) {
                              for (auto& i : label_indices) {
                                  image.flat(i) == 0;
                              }
                          } else if (photons == 1) {
                              // Find coordinates of the brightest pixels
                              std::array<size_t, 2> sp_idx{ idxs_weighted_mean(label_indices) };

                              // Zero the droplet
                              for (auto& i : label_indices) {
                                  image.flat(i) = 0;
                              }

                              // Assign photon
                              photon_data[sp_idx] += 1;
                          } else {
                              // Iterate over the remaining photons
                              for (int i = 0; i < photons; ++i) {
                                  // Find the brightest one
                                  int max_idx{ 0 };
                                  for (auto& idx : label_indices) {
                                      if (image.flat(idx) > image.flat(max_idx)) {
                                          max_idx = idx;
                                      }
                                  }

                                  if (image.flat(max_idx) == 0) {
                                      continue;
                                  }

                                  // If this is the brightest, add it to the photon counts
                                  if (image.flat(max_idx) >= adu_count) {
                                      photon_data.flat(max_idx) += 1;
                                      image.flat(max_idx) -= adu_count;
                                  } else {
                                      // Otherwise, look for the second brightest around the
                                      // neighbours.
                                      size_t max_neighbour_idx{ 0 };
                                      auto max_array_idx{ xt::unravel_index(max_idx, image.shape()) };
                                      for (auto offset : std::initializer_list<std::array<int, 2>>({{0, -1}, {0, 1}, {1, 0}, {-1, 0}})) {
                                          std::array<long, 2> neighbour_idx({ max_array_idx[0] + offset[0],
                                                                                max_array_idx[1] + offset[1] });

                                          if (image[neighbour_idx] > image.flat(max_neighbour_idx)) {
                                              max_neighbour_idx = xt::ravel_index(neighbour_idx, image.shape(), image.layout());
                                          }
                                      }

                                      if (image.flat(max_neighbour_idx) == 0) {
                                          continue;
                                      }

                                      // Decide if the second photon is in the brightest
                                      // pixel or the neighbour.
                                      float distance{ (adu_count - image.flat(max_idx)) / adu_count };
                                      photon_data.flat(distance <= 0.5 ? max_idx : max_neighbour_idx) += 1;
                                      image.flat(max_idx) = 0;
                                      image.flat(max_neighbour_idx) = 0;
                                  }
                              }
                          }
                      });
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
