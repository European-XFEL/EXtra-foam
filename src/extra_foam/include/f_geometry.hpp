/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#ifndef EXTRA_FOAM_GEOMETRY_H
#define EXTRA_FOAM_GEOMETRY_H

#include <cassert>
#include <cmath>
#include <array>
#include <type_traits>
#include <algorithm>

#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xindex_view.hpp"
#if defined(FOAM_USE_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#endif

#include "f_traits.hpp"
#include <algorithm>

namespace foam
{

struct DetectorBase
{
  using VectorType = xt::xtensor_fixed<double, xt::xshape<3>>;
  using ShapeType = std::array<size_t, 2>;
};

struct JungFrau : public DetectorBase
{
  static const ShapeType module_shape;
  static const ShapeType asic_shape;
  static const VectorType pixel_size;
  static const ShapeType asic_grid_shape;
};

const JungFrau::ShapeType JungFrau::module_shape {512, 1024}; // (y, x)
const JungFrau::ShapeType JungFrau::asic_shape {256, 256}; // (y, x)
const JungFrau::VectorType JungFrau::pixel_size {75e-6, 75e-6, 1.}; // (x, y, z)
const JungFrau::ShapeType JungFrau::asic_grid_shape {2, 4}; // (y, x)

struct EPix100 : public DetectorBase
{
  static const ShapeType module_shape;
  static const ShapeType asic_shape;
  static const VectorType pixel_size;
  static const ShapeType asic_grid_shape;
};

const EPix100::ShapeType EPix100::module_shape {708, 768}; // (y, x)
// TODO: in the paper it says 352 x 384
const EPix100::ShapeType EPix100::asic_shape {354, 384}; // (y, x)
const EPix100::VectorType EPix100::pixel_size {50e-6, 50e-6, 1.}; // (x, y, z)
const EPix100::ShapeType EPix100::asic_grid_shape {2, 2}; // (y, x)

enum class GeometryLayout
{
  TopRightCW = 0x01,
  BottomRightCCW = 0x01,
  BottomLeftCW = 0x03,
  TopLeftCCW = 0x04,
};

/**
 * Base class for generalized detector geometry. For detectors like JungFrau,
 * it is possible to use a single module or the combination of multiple modules
 * during experiments. Unlike the 1M detectors, the number of modules is variable
 * and the way of stacking them is still unclear. On the other hand, each module
 * is consisting of multiple ASICs and there is no tile.
 *
 * When more than one modules are combined, the layout of modules is defined by
 * the enum class GeometryLayout:
 *
 *  - TopRightCW: the first module starts at top-right, the modules are
 *                    arranged in clockwise order. For example:
 *
 *                    M4 | M1
 *                    -------
 *                    M3 | M2
 *
 *  - BottomRightCCW: the first module starts at bottom-right, the modules are
 *                    arranged in counter clockwise order. For example:
 *
 *                    M3 | M2
 *                    -------
 *                    M4 | M1
 *
 *  - BottomLeftCW: the first module starts at bottom-left, the modules are
 *                  arranged in clockwise order. For example:
 *
 *                    M2 | M3
 *                    -------
 *                    M1 | M4
 *
 *  - TopLeftCCW: the first module starts at top-left, the modules are
 *                    arranged in counter clockwise order. For example:
 *
 *                    M1 | M4
 *                    -------
 *                    M2 | M3
 *
 *  Since module data is passed by an array, there is no information about the
 *  real index of the module. Therefore, the modules are arranged by their
 *  indices. One can achieve the following layout simply by passing an array-like
 *  data as [M1, M2, M3, M6, M7, M8].
 *
 *                    M6 | M1
 *                    -------
 *                    M5 | M2
 *                    -------
 *                    M4 | M3
 *
 */
template<typename Detector>
class DetectorGeometry
{
public:

  using ShapeType = typename Detector::ShapeType;
  using CenterType = std::array<int, 2>;
  using VectorType = typename Detector::VectorType;
  using OrientationType = xt::xtensor<double, 2>;
  using PositionType = xt::xtensor<double, 2>;

private:

  size_t n_rows_; // number of rows
  size_t n_columns_; // number of columns
  size_t n_modules_; // total number of modules

  GeometryLayout layout_type_;
  OrientationType orients_;
  std::pair<PositionType, PositionType> corner_pos_;
  ShapeType a_shape_;
  CenterType a_center_;

public:

  DetectorGeometry(size_t n_rows, size_t n_columns,
                   GeometryLayout layout_type = GeometryLayout::TopRightCW);

  DetectorGeometry(size_t n_rows, size_t n_columns,
                   const std::vector<std::array<double, 3>>& positions,
                   GeometryLayout layout_type = GeometryLayout::TopRightCW);

  ~DetectorGeometry() = default;

  /**
   * Position all the modules at the correct area of the given assembled image.
   *
   * @param src: data in modules. shape=(modules, y, x)
   * @param dst: assembled image. shape=(y, x)
   * @param ignore_asic_edge: true for ignoring the pixels at the edges of asics. If dst
   *    is pre-filled with nan, it it equivalent to masking the asic edges.
   */
  template<typename M, typename E,
    EnableIf<std::decay_t<M>, IsImageArray> = false, EnableIf<E, IsImage> = false>
  void positionAllModules(M&& src, E& dst, bool ignore_asic_edge=false) const;

  /**
   * Position all the modules at the correct area of the given assembled image.
   *
   * @param src: a vector of modules data, which has a shape of (y, x)
   * @param dst: assembled image. shape=(y, x)
   * @param ignore_asic_edge: true for ignoring the pixels at the edges of asics. If dst
   *    is pre-filled with nan, it it equivalent to masking the asic edges.
   */
  template<typename M, typename E,
    EnableIf<std::decay_t<M>, IsImageVector> = false, EnableIf<E, IsImage> = false>
  void positionAllModules(M&& src, E& dst, bool ignore_asic_edge=false) const;

  /**
   * Position all the modules at the correct area of the given assembled image.
   *
   * @param src: multi-pulse, multiple-module data. shape=(memory cells, modules, y, x)
   * @param dst: assembled data. shape=(memory cells, y, x)
   * @param ignore_asic_edge: true for ignoring the pixels at the edges of asics. If dst
   *    is pre-filled with nan, it it equivalent to masking the asic edges.
   */
  template<typename M, typename E,
    EnableIf<std::decay_t<M>, IsModulesArray> = false, EnableIf<E, IsImageArray> = false>
  void positionAllModules(M&& src, E& dst, bool ignore_asic_edge=false) const;

  /**
   * Position all the modules at the correct area of the given assembled image.
   *
   * @param src: a vector of module data, which has a shape of (modules, y, x)
   * @param dst: assembled data. shape=(memory cells, y, x)
   * @param ignore_asic_edge: true for ignoring the pixels at the edges of asics. If dst
   *    is pre-filled with nan, it it equivalent to masking the asic edges.
   */
  template<typename M, typename E,
    EnableIf<std::decay_t<M>, IsModulesVector> = false, EnableIf<E, IsImageArray> = false>
  void positionAllModules(M&& src, E& dst, bool ignore_asic_edge=false) const;

  template<typename M, EnableIf<M, IsImage> = false>
  static void maskModule(M& src);

  template<typename M, EnableIf<M, IsImageArray> = false>
  static void maskModule(M& src);

  /**
   * Dismantle an assembled image into modules.
   *
   * @param src: assembled data (y, x)
   * @param dst: data in modules. shape=(modules, y, x)
   */
  template<typename M, typename E,
    EnableIf<std::decay_t<M>, IsImage> = false, EnableIf<E, IsImageArray> = false>
  void dismantleAllModules(M&& src, E& dst) const;

  /**
   * Dismantle all assembled images into modules.
   *
   * @param src: assembled data (memory cells, y, x)
   * @param dst: data in modules. shape=(memory cells, modules, y, x)
   */
  template<typename M, typename E,
    EnableIf<std::decay_t<M>, IsImageArray> = false, EnableIf<E, IsModulesArray> = false>
  void dismantleAllModules(M&& src, E& dst) const;

  /**
   * Return the shape (y, x) of the assembled image.
   */
  const ShapeType& assembledShape() const;

  /**
   * Return the center (x, y) of the assembled image.
   */
  const CenterType& assembledCenter() const;

  /**
   * Return the number of modules.
   */
  size_t nModules() const;

  /**
   * Return the shape (y, x) of a module.
   */
  const ShapeType& moduleShape() const;

  /**
   * Return the shape (y, x) of a ASIC.
   */
  const ShapeType& asicShape() const;

private:

  /**
   * Initialize module origins based on numbers of rows and columns as well as the layout type.
   */
  void initModuleOrigins(bool stack_only=false);

  /**
   * Initialize module orientations based on numbers of rows and columns as well as the layout type.
   */
  void initModuleOrientations();

  /**
   * Calculate the size (y, x) and center (x, y) of the assembled image.
   */
  void computeAssembledDim();

  /**
   * Check the src and dst shapes used for assembling.
   *
   * @param ss: src data shape (memory cells, modules, y, x).
   * @param ds: dst data shape (memory cells, y, x)
   */
  template<typename SrcShape, typename DstShape>
  void checkShapeForAssembling(const SrcShape& ss, const DstShape& ds) const;

  /**
   * Position a single module at the assembled image.
   *
   * @param src: data from a single module. shape=(y, x).
   * @param dst: assembled single image. shape=(y, x)
   * @param p0: two diagonal corner positions of each tile. shape=(3,)
   * @param p1: two diagonal corner positions of each tile. shape=(3,)
   * @param ignore_asic_edge: true for ignoring the pixels at the edges of asics. If dst
   *   is pre-filled with nan, it it equivalent to masking the asic edges.
   */
  template<typename M,  typename N, typename T>
  void positionModule(M&& src, N& dst, T&& p0, T&& p1, bool ignore_asic_edge) const;

  template<typename M>
  static void maskModuleImp(M& src);

  /**
   * Check the src and dst shapes used for dismantling..
   *
   * @param ss: src data shape (memory cells, y, x).
   * @param ds: dst data shape (memory cells, modules, y, x)
   */
  template<typename SrcShape, typename DstShape>
  void checkShapeForDismantling(const SrcShape& ss, const DstShape& ds) const;

  /**
   * Dismantle a single module into tiles.
   *
   * @param src: data from a single module. shape=(y, x).
   * @param dst: assembled single image. shape=(y, x)
   * @param p0: two diagonal corner positions of each tile. shape=(3,)
   * @param p1: two diagonal corner positions of each tile. shape=(3,)
   */
  template<typename M, typename N, typename T>
  void dismantleModule(M&& src, N& dst, T&& p0, T&& p1) const;

};

template<typename Detector>
DetectorGeometry<Detector>::DetectorGeometry(size_t n_rows,
                                             size_t n_columns,
                                             GeometryLayout layout_type)
  : n_rows_(n_rows), n_columns_(n_columns), n_modules_(n_rows * n_columns), layout_type_(layout_type)
{
  initModuleOrigins(true);
  initModuleOrientations();

  // first pixel position of each module
  auto w = static_cast<double>(Detector::module_shape[1]);
  auto h = static_cast<double>(Detector::module_shape[0]);

  for (size_t im = 0; im < n_modules_; ++im)
  {
    // calculate the position of the diagonal corner
    corner_pos_.second(im, 0) = corner_pos_.first(im, 0) + w * orients_(im, 0);
    corner_pos_.second(im, 1) = corner_pos_.first(im, 1) + h * orients_(im, 1);
    corner_pos_.second(im, 2) = corner_pos_.first(im, 2);
  }

  corner_pos_.first *= Detector::pixel_size;
  corner_pos_.second *= Detector::pixel_size;

  computeAssembledDim();
}

template<typename Detector>
DetectorGeometry<Detector>::DetectorGeometry(size_t n_rows, size_t n_columns,
                                             const std::vector<std::array<double, 3>>& positions,
                                             GeometryLayout layout_type)
  : n_rows_(n_rows), n_columns_(n_columns), n_modules_(n_rows * n_columns), layout_type_(layout_type)
{
  initModuleOrigins();
  initModuleOrientations();

  auto w = static_cast<double>(Detector::module_shape[1]);
  auto h = static_cast<double>(Detector::module_shape[0]);

  for (size_t im = 0; im < n_modules_; ++im)
  {
    for (int j = 0; j < 3; ++j) corner_pos_.first(im, j) = positions[im][j];
    // calculate the position of the diagonal corner
    corner_pos_.second(im, 0) = positions[im][0] + w * Detector::pixel_size(0) * orients_(im, 0);
    corner_pos_.second(im, 1) = positions[im][1] + h * Detector::pixel_size(1) * orients_(im, 1);
    corner_pos_.second(im, 2) = positions[im][2];
  }

  computeAssembledDim();
}

template<typename Detector>
template<typename M, typename E, EnableIf<std::decay_t<M>, IsImageArray>, EnableIf<E, IsImage>>
void DetectorGeometry<Detector>::positionAllModules(M&& src, E& dst, bool ignore_asic_edge) const
{
  auto ss = src.shape();
  auto ds = dst.shape();
  // the shape dtype of xt::pytensor is npy_intp
  checkShapeForAssembling(std::array<size_t, 4>({1, static_cast<size_t>(ss[0]), static_cast<size_t>(ss[1]), static_cast<size_t>(ss[2])}),
                          std::array<size_t, 3>({1, static_cast<size_t>(ds[0]), static_cast<size_t>(ds[1])}));

  auto p0 = corner_pos_.first / Detector::pixel_size;
  auto p1 = corner_pos_.second / Detector::pixel_size;

  for (size_t im = 0; im < n_modules_; ++im)
  {
    positionModule(
      xt::view(src, im, xt::all(), xt::all()),
      dst,
      xt::view(p0, im, xt::all(), xt::all()),
      xt::view(p1, im, xt::all(), xt::all()),
      ignore_asic_edge
    );
  }
}

template<typename Detector>
template<typename M, typename E, EnableIf<std::decay_t<M>, IsImageVector>, EnableIf<E, IsImage>>
void DetectorGeometry<Detector>::positionAllModules(M&& src, E& dst, bool ignore_asic_edge) const
{

  auto ms = src[0].shape();
  auto ds = dst.shape();
  // the shape dtype of xt::pytensor is npy_intp
  this->checkShapeForAssembling(std::array<size_t, 4>({1, src.size(), static_cast<size_t>(ms[0]), static_cast<size_t>(ms[1])}),
                                std::array<size_t, 3>({1, static_cast<size_t>(ds[0]), static_cast<size_t>(ds[1])}));

  auto p0 = corner_pos_.first / Detector::pixel_size;
  auto p1 = corner_pos_.second / Detector::pixel_size;

  for (size_t im = 0; im < n_modules_; ++im)
  {
    positionModule(
      src[im],
      dst,
      xt::view(p0, im, xt::all(), xt::all()),
      xt::view(p1, im, xt::all(), xt::all()),
      ignore_asic_edge
    );
  }
}

template<typename Detector>
template<typename M, typename E, EnableIf<std::decay_t<M>, IsModulesArray>, EnableIf<E, IsImageArray>>
void DetectorGeometry<Detector>::positionAllModules(M&& src, E& dst, bool ignore_asic_edge) const
{
  auto ss = src.shape();
  auto ds = dst.shape();
  this->checkShapeForAssembling(ss, ds);

  size_t n_pulses = ss[0];
  auto p0 = corner_pos_.first / Detector::pixel_size;
  auto p1 = corner_pos_.second / Detector::pixel_size;
#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range2d<int>(0, n_modules_, 0, n_pulses),
    [&src, &dst, &p0, &p1, ignore_asic_edge, this] (const tbb::blocked_range2d<int> &block)
    {
      for(int im=block.rows().begin(); im != block.rows().end(); ++im)
      {
        for(int ip=block.cols().begin(); ip != block.cols().end(); ++ip)
        {
#else
      for (size_t im = 0; im < n_modules_; ++im)
        {
        for (size_t ip = 0; ip < n_pulses; ++ip)
        {
#endif
          auto&& dst_view = xt::view(dst, ip, xt::all(), xt::all());
          positionModule(
            xt::view(src, ip, im, xt::all(), xt::all()),
            dst_view,
            xt::view(p0, im, xt::all(), xt::all()),
            xt::view(p1, im, xt::all(), xt::all()),
            ignore_asic_edge);
        }
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif
}

template<typename Detector>
template<typename M, typename E, EnableIf<std::decay_t<M>, IsModulesVector>, EnableIf<E, IsImageArray>>
void DetectorGeometry<Detector>::positionAllModules(M&& src, E& dst, bool ignore_asic_edge) const
{
  auto ms = src[0].shape();
  // the shape dtype of xt::pytensor is npy_intp
  auto ss = std::array<size_t, 4> { static_cast<size_t>(ms[0]), src.size(), static_cast<size_t>(ms[1]), static_cast<size_t>(ms[2]) };
  auto ds = dst.shape();
  this->checkShapeForAssembling(ss, ds);

  size_t n_pulses = ss[0];
  auto p0 = corner_pos_.first / Detector::pixel_size;
  auto p1 = corner_pos_.second / Detector::pixel_size;
#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range2d<int>(0, n_modules_, 0, n_pulses),
    [&src, &dst, &p0, &p1, ignore_asic_edge, this] (const tbb::blocked_range2d<int> &block)
    {
      for(int im=block.rows().begin(); im != block.rows().end(); ++im)
      {
        for(int ip=block.cols().begin(); ip != block.cols().end(); ++ip)
        {
#else
      for (size_t im = 0; im < n_modules_; ++im)
      {
        for (size_t ip = 0; ip < n_pulses; ++ip)
        {
#endif
          auto&& dst_view = xt::view(dst, ip, xt::all(), xt::all());
          positionModule(
            xt::view(src[im], ip, xt::all(), xt::all()),
            dst_view,
            xt::view(p0, im, xt::all(), xt::all()),
            xt::view(p1, im, xt::all(), xt::all()),
            ignore_asic_edge);
        }
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif
}

template<typename Detector>
template<typename M, typename E, EnableIf<std::decay_t<M>, IsImage>, EnableIf<E, IsImageArray>>
void DetectorGeometry<Detector>::dismantleAllModules(M&& src, E& dst) const
{
  auto ss = src.shape();
  auto ds = dst.shape();
  // the shape dtype of xt::pytensor is npy_intp
  checkShapeForDismantling(std::array<size_t, 3>({1, static_cast<size_t>(ss[0]), static_cast<size_t>(ss[1])}),
                           std::array<size_t, 4>({1, static_cast<size_t>(ds[0]), static_cast<size_t>(ds[1]), static_cast<size_t>(ds[2])}));

  auto p0 = corner_pos_.first / Detector::pixel_size;
  auto p1 = corner_pos_.second / Detector::pixel_size;
  for (size_t im = 0; im < n_modules_; ++im)
  {
    auto&& dst_view = xt::view(dst, im, xt::all(), xt::all());
    dismantleModule(
      src,
      dst_view,
      xt::view(p0, im, xt::all(), xt::all()),
      xt::view(p1, im, xt::all(), xt::all()));
  }
}

template<typename Detector>
template<typename M, typename E, EnableIf<std::decay_t<M>, IsImageArray>, EnableIf<E, IsModulesArray>>
void DetectorGeometry<Detector>::dismantleAllModules(M&& src, E& dst) const
{
  auto ss = src.shape();
  auto ds = dst.shape();
  checkShapeForDismantling(ss, ds);

  size_t n_pulses = ss[0];
  auto p0 = corner_pos_.first / Detector::pixel_size;
  auto p1 = corner_pos_.second / Detector::pixel_size;
#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range2d<int>(0, n_modules_, 0, n_pulses),
    [&src, &dst, &p0, &p1, this] (const tbb::blocked_range2d<int> &block)
    {
      for(int im=block.rows().begin(); im != block.rows().end(); ++im)
      {
        for(int ip=block.cols().begin(); ip != block.cols().end(); ++ip)
        {
#else
      for (size_t im = 0; im < n_modules_; ++im)
      {
        for (size_t ip = 0; ip < n_pulses; ++ip)
        {
#endif
          auto&& dst_view = xt::view(dst, ip, im, xt::all(), xt::all());
          dismantleModule(
            xt::view(src, ip, xt::all(), xt::all()),
            dst_view,
            xt::view(p0, im, xt::all(), xt::all()),
            xt::view(p1, im, xt::all(), xt::all()));
        }
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif
}

template<typename Detector>
void DetectorGeometry<Detector>::initModuleOrigins(bool stack_only)
{
  if (n_columns_ > 2) throw std::invalid_argument("Number of columns can be either 1 or 2!");

  corner_pos_.first = PositionType::from_shape({n_modules_, 3});
  corner_pos_.second = PositionType::from_shape({n_modules_, 3});

  if (!stack_only) return;

  auto w = static_cast<double>(Detector::module_shape[1]);
  auto h = static_cast<double>(Detector::module_shape[0]);
  auto n_modules = static_cast<int>(n_modules_);
  if (layout_type_ == GeometryLayout::TopRightCW)
  {
    for (int nm = 0; nm < n_modules; ++nm)
    {
      if (nm < static_cast<int>(n_rows_))
      {
        corner_pos_.first(nm, 0) = w;
        corner_pos_.first(nm, 1) = - h * nm;
      } else
      {
        corner_pos_.first(nm, 0) = -w;
        corner_pos_.first(nm, 1) = - h * (n_modules - nm);
      }
      corner_pos_.first(nm, 2) = 0;
    }
  } else
  {
    throw std::runtime_error("Not implemented!");
  }
}

template<typename Detector>
void DetectorGeometry<Detector>::initModuleOrientations()
{
  orients_ = PositionType::from_shape({n_modules_, 2});

  if (layout_type_ == GeometryLayout::TopRightCW)
  {
    for (size_t nm = 0; nm < n_modules_; ++nm)
    {
      orients_(nm, 0) = nm / n_rows_ == 0 ? -1 : 1;
      orients_(nm, 1) = nm / n_rows_ == 0 ? -1 : 1;
    }
  } else
  {
    throw std::runtime_error("Not implemented!");
  }

}

template<typename Detector>
const typename DetectorGeometry<Detector>::ShapeType& DetectorGeometry<Detector>::assembledShape() const
{
  return a_shape_;
}

template<typename Detector>
const typename DetectorGeometry<Detector>::CenterType& DetectorGeometry<Detector>::assembledCenter() const
{
  return a_center_;
}

template<typename Detector>
size_t DetectorGeometry<Detector>::nModules() const
{
  return n_modules_;
}

template<typename Detector>
const typename DetectorGeometry<Detector>::ShapeType& DetectorGeometry<Detector>::moduleShape() const
{
  return Detector::module_shape;
}

template<typename Detector>
const typename DetectorGeometry<Detector>::ShapeType& DetectorGeometry<Detector>::asicShape() const
{
  return Detector::asic_shape;
}

template<typename Detector>
void DetectorGeometry<Detector>::computeAssembledDim()
{
  auto min_xyz_0 = xt::amin(corner_pos_.first, {0}) / Detector::pixel_size;
  auto max_xyz_0 = xt::amax(corner_pos_.first, {0}) / Detector::pixel_size;
  auto min_xyz_1 = xt::amin(corner_pos_.second, {0}) / Detector::pixel_size;
  auto max_xyz_1 = xt::amax(corner_pos_.second, {0}) / Detector::pixel_size;

  auto min_x = static_cast<int>(std::round(std::min(min_xyz_0[0], min_xyz_1[0])));
  auto min_y = static_cast<int>(std::round(std::min(min_xyz_0[1], min_xyz_1[1])));
  auto max_x = static_cast<int>(std::round(std::max(max_xyz_0[0], max_xyz_1[0])));
  auto max_y = static_cast<int>(std::round(std::max(max_xyz_0[1], max_xyz_1[1])));

  a_shape_ = { static_cast<size_t>(max_y - min_y), static_cast<size_t>(max_x - min_x) };
  a_center_ = {-min_x, -min_y};
}

template<typename Detector>
template<typename SrcShape, typename DstShape>
void DetectorGeometry<Detector>::checkShapeForAssembling(const SrcShape& ss, const DstShape& ds) const
{
  if (ss[0] != ds[0])
  {
    std::stringstream fmt;
    fmt << "Modules data and output array have different memory cells: "
        << ss[0] << " and " << ds[0] << "!";
    throw std::invalid_argument(fmt.str());
  }

  if (ss[1] != n_modules_)
  {
    std::stringstream fmt;
    fmt << "Expected " << n_modules_ << " modules, get " << ss[1] << "!";
    throw std::invalid_argument(fmt.str());
  }

  if ( (ss[2] != Detector::module_shape[0]) || (ss[3] != Detector::module_shape[1]) )
  {
    std::stringstream fmt;
    fmt << "Expected modules with shape (" << Detector::module_shape[0] << ", " << Detector::module_shape[1]
        << ") modules, get (" << ss[2] << ", " << ss[3] << ")!";
    throw std::invalid_argument(fmt.str());
  }

  auto as = assembledShape();
  if ( (as[0] != ds[1]) | (as[1] != ds[2]) )
  {
    std::stringstream fmt;
    fmt << "Expected output array with shape (" << as[0] << ", " << as[1]
        << ") modules, get (" << ds[1] << ", " << ds[2] << ")!";
    throw std::invalid_argument(fmt.str());
  }
}

template<typename Detector>
template<typename M, typename N, typename T>
void DetectorGeometry<Detector>::positionModule(M&& src, N& dst, T&& p0, T&& p1, bool ignore_asic_edge) const
{
  int wa = Detector::asic_shape[1];
  int ha = Detector::asic_shape[0];

  int edge = 0;
  if (ignore_asic_edge) edge = 1;

  double x0 = p0(0);
  double y0 = p0(1);

  int ix_dir = (p1(0) - x0 > 0) ? 1 : -1;
  int iy_dir = (p1(1) - y0 > 0) ? 1 : -1;

  for (int i_row = 0; i_row < static_cast<int>(Detector::asic_grid_shape[0]); ++i_row)
  {
    for (int i_col = 0; i_col < static_cast<int>(Detector::asic_grid_shape[1]); ++i_col)
    {
      int ix0 = wa * i_col;
      int iy0 = ha * i_row;

      int ix0_dst = static_cast<int>(std::round(x0)) + a_center_[0] + ix_dir * wa * i_col;
      if (ix_dir < 0) --ix0_dst;
      int iy0_dst = static_cast<int>(std::round(y0)) + a_center_[1] + iy_dir * ha * i_row;
      if (iy_dir < 0) --iy0_dst;

      for (int iy = iy0 + edge, iy_dst = iy0_dst + iy_dir * edge; iy < ha * i_row + ha - edge; ++iy, iy_dst += iy_dir)
      {
        for (int ix = ix0 + edge, ix_dst = ix0_dst + ix_dir * edge; ix < wa * (i_col + 1) - edge; ++ix, ix_dst += ix_dir)
        {
          dst(iy_dst, ix_dst) = src(iy, ix);
        }
      }
    }
  }
}

template<>
template<typename M, typename N, typename T>
void DetectorGeometry<EPix100>::positionModule(M&& src, N& dst, T&& p0, T&& p1, bool ignore_asic_edge) const
{
  int wm = EPix100::module_shape[1];
  int hm = EPix100::module_shape[0];

  int edge = 0;
  if (ignore_asic_edge) edge = 1;

  double x0 = p0(0);
  double y0 = p0(1);

  int ix_dir = (p1(0) - x0 > 0) ? 1 : -1;
  int iy_dir = (p1(1) - y0 > 0) ? 1 : -1;

  int ix0 = 0;
  int iy0 = 0;

  int ix0_dst = static_cast<int>(std::round(x0)) + a_center_[0];
  if (ix_dir < 0) --ix0_dst;
  int iy0_dst = static_cast<int>(std::round(y0)) + a_center_[1];
  if (iy_dir < 0) --iy0_dst;

  for (int iy = iy0 + edge, iy_dst = iy0_dst + iy_dir * edge; iy < hm - edge; ++iy, iy_dst += iy_dir)
  {
    for (int ix = ix0, ix_dst = ix0_dst; ix < wm; ++ix, ix_dst += ix_dir)
    {
      dst(iy_dst, ix_dst) = src(iy, ix);
    }
  }
}

template<typename Detector>
template<typename M>
void DetectorGeometry<Detector>::maskModuleImp(M& src)
{
  auto ss = src.shape();
  if ( (ss[1] != Detector::module_shape[0]) || (ss[2] != Detector::module_shape[1]) )
  {
    std::stringstream fmt;
    fmt << "Expected module with shape (" << Detector::module_shape[0] << ", " << Detector::module_shape[1]
        << ") modules, get (" << ss[1] << ", " << ss[2] << ")!";
    throw std::invalid_argument(fmt.str());
  }

  int wa = Detector::asic_shape[1];
  int ha = Detector::asic_shape[0];

  auto nan = std::numeric_limits<typename M::value_type>::quiet_NaN();

  for (size_t i_row = 0; i_row < Detector::asic_grid_shape[0]; ++i_row)
  {
    xt::view(src, xt::all(), i_row * ha, xt::all()) = nan;
    xt::view(src, xt::all(), (i_row + 1) * ha - 1, xt::all()) = nan;
  }
  for (size_t i_col = 0; i_col < Detector::asic_grid_shape[1]; ++i_col)
  {
    xt::view(src, xt::all(), xt::all(), i_col * wa) = nan;
    xt::view(src, xt::all(), xt::all(), (i_col + 1) * wa - 1) = nan;
  }
}

template<>
template<typename M>
void DetectorGeometry<EPix100>::maskModuleImp(M& src)
{
  auto ss = src.shape();
  if ( (ss[1] != EPix100::module_shape[0]) || (ss[2] != EPix100::module_shape[1]) )
  {
    std::stringstream fmt;
    fmt << "Expected module with shape (" << EPix100::module_shape[0] << ", " << EPix100::module_shape[1]
        << ") modules, get (" << ss[1] << ", " << ss[2] << ")!";
    throw std::invalid_argument(fmt.str());
  }

  auto nan = std::numeric_limits<typename M::value_type>::quiet_NaN();

  xt::view(src, xt::all(), 0, xt::all()) = nan;
  xt::view(src, xt::all(), EPix100::module_shape[0] - 1, xt::all()) = nan;
}

template<typename Detector>
template<typename M, EnableIf<M, IsImageArray>>
void DetectorGeometry<Detector>::maskModule(M& src)
{
  maskModuleImp(src);
}

template<typename Detector>
template<typename M, EnableIf<M, IsImage>>
void DetectorGeometry<Detector>::maskModule(M& src)
{
  auto&& expanded = xt::view(src, xt::newaxis(), xt::all(), xt::all());
  maskModuleImp(expanded);
}

template<typename Detector>
template<typename SrcShape, typename DstShape>
void DetectorGeometry<Detector>::checkShapeForDismantling(const SrcShape& ss, const DstShape& ds) const
{

  if (ss[0] != ds[0])
  {
    std::stringstream fmt;
    fmt << "Input and output array have different memory cells: " << ss[0] << " and " << ds[0] << "!";
    throw std::invalid_argument(fmt.str());
  }

  auto expected_ss = assembledShape();
  if ( (ss[1] != expected_ss[0]) || (ss[2] != expected_ss[1]) )
  {
    std::stringstream fmt;
    fmt << "Expected source image with shape (" << expected_ss[0] << ", " << expected_ss[1]
        << ") modules, get (" << ss[1] << ", " << ss[2] << ")!";
    throw std::invalid_argument(fmt.str());
  }

  if ( (ds[1] != n_modules_) | (ds[2] != Detector::module_shape[0]) | (ds[3] != Detector::module_shape[1]) )
  {
    std::stringstream fmt;
    fmt << "Expected output array with shape ("
        << n_modules_ << ", " << Detector::module_shape[0] << ", " << Detector::module_shape[1] << ") modules, get ("
        << ds[1] << ", " << ds[2] << ", " << ds[3] << ")!";
    throw std::invalid_argument(fmt.str());
  }
}

template<typename Detector>
template<typename M, typename N, typename T>
void DetectorGeometry<Detector>::dismantleModule(M&& src, N& dst, T&& p0, T&& p1) const
{
  int wm = Detector::module_shape[1];
  int hm = Detector::module_shape[0];

  double x0 = p0(0);
  double y0 = p0(1);

  int ix_dir = (p1(0) - x0 > 0) ? 1 : -1;
  int iy_dir = (p1(1) - y0 > 0) ? 1 : -1;

  size_t ix0 = static_cast<int>(std::round(x0)) + a_center_[0];
  if (ix_dir < 0) --ix0;
  size_t iy0 = static_cast<int>(std::round(y0)) + a_center_[1];
  if (iy_dir < 0) --iy0;

  for (int iy = iy0, iy_dst = 0; iy_dst < hm; ++iy_dst, iy += iy_dir)
  {
    for (int ix = ix0, ix_dst = 0; ix_dst < wm; ++ix_dst, ix += ix_dir)
    {
      dst(iy_dst, ix_dst) = src(iy, ix);
    }
  }
}

} // foam

#endif //EXTRA_FOAM_GEOMETRY_H
