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
#if defined(FOAM_WITH_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#include "tbb/blocked_range3d.h"
#endif

#include "f_traits.hpp"
#include <algorithm>

namespace foam
{

template<typename G>
class Detector1MGeometryBase
{

public:

  static constexpr int n_quads = 4;
  static constexpr int n_modules_per_quad = 4;
  static constexpr int n_modules = n_quads * n_modules_per_quad;

  using vectorType = xt::xtensor_fixed<double, xt::xshape<3>>;
  using quadOrientType = std::array<std::array<int, 2>, 4>;
  using quadVectorType = xt::xtensor_fixed<double, xt::xshape<n_quads, 3>>;
  using shapeType = std::array<int, 2>;
  using vector2dType = std::array<double, 2>;

  ~Detector1MGeometryBase() = default;

  /**
   * Position all the modules at the correct area of the given assembled image.
   *
   * @param src: multi-pulse, multiple-module data. shape=(memory cells, modules, y, x)
   * @param dst: assembled data. shape=(memory cells, y, x)
   * @param ignore_tile_edge: true for ignoring the pixels at the edges of tiles. If dst
   *    is pre-filled with nan, it it equivalent to masking the tile edges.
   */
  template<typename M, typename E,
    EnableIf<std::decay_t<M>, IsModulesArray> = false, EnableIf<E, IsImageArray> = false>
  void positionAllModules(M&& src, E& dst, bool ignore_tile_edge=false) const;

  /**
   * Position all the modules at the correct area of the given assembled image.
   *
   * @param src: a vector of module data, which has a shape of (modules, y, x)
   * @param dst: assembled data. shape=(memory cells, y, x)
   * @param ignore_tile_edge: true for ignoring the pixels at the edges of tiles. If dst
   *    is pre-filled with nan, it it equivalent to masking the tile edges.
   */
  template<typename M, typename E,
    EnableIf<std::decay_t<M>, IsModulesVector> = false, EnableIf<E, IsImageArray> = false>
  void positionAllModules(M&& src, E& dst, bool ignore_tile_edge=false) const;

  /**
   * Return the shape (y, x) of the assembled image.
   */
  shapeType assembledShape() const
  {
    auto size = assembledDim().first;
    return {static_cast<int>(std::ceil(size[0])), static_cast<int>(std::ceil(size[1]))};
  }

protected:

  Detector1MGeometryBase() = default;

  /**
   * Return the size (y, x) and center (x, y) of the assembled image.
   */
  std::pair<vector2dType, vector2dType> assembledDim() const;

  /**
   * Check the src and dst shapes.
   *
   * @param ss: src data shape (memory cells, modules, y, x).
   * @param ds: dst data shape (memory cells, y, x)
   */
  template<typename SrcShape, typename DstShape>
  void checkShape(const SrcShape& ss, const DstShape& ds) const;

  /**
   * Position a single module at the assembled image.
   *
   * @param src: data from a single module. shape=(memory cells, y, x).
   * @param dst: assembled single image. shape=(y, x)
   * @param pos: two diagonal corner positions of each tile. shape=(tiles, 2, 3)
   */
  template<typename M, typename N, typename T>
  void positionModule(M&& src, N& dst, T&& pos, bool ignore_tile_edge) const;
};

template<typename G>
constexpr int Detector1MGeometryBase<G>::n_quads;
template<typename G>
constexpr int Detector1MGeometryBase<G>::n_modules_per_quad;
template<typename G>
constexpr int Detector1MGeometryBase<G>::n_modules;

template<typename G>
template<typename M, typename E, EnableIf<std::decay_t<M>, IsModulesArray>, EnableIf<E, IsImageArray>>
void Detector1MGeometryBase<G>::positionAllModules(M&& src, E& dst, bool ignore_tile_edge) const
{
  auto ss = src.shape();
  auto ds = dst.shape();
  this->checkShape(ss, ds);

  int n_pulses = ss[0];
  auto norm_pos = static_cast<const G*>(this)->corner_pos_ / static_cast<const G*>(this)->pixelSize();
#if defined(FOAM_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range2d<int>(0, n_modules, 0, n_pulses),
    [&src, &dst, &norm_pos, ignore_tile_edge, this] (const tbb::blocked_range2d<int> &block)
    {
      for(int im=block.rows().begin(); im != block.rows().end(); ++im)
      {
        for(int ip=block.cols().begin(); ip != block.cols().end(); ++ip)
        {
#else
      for (int im = 0; im < n_modules; ++im)
        {
        for (int ip = 0; ip < n_pulses; ++ip)
        {
#endif
          auto&& dst_view = xt::view(dst, ip, xt::all(), xt::all());
          positionModule(
            xt::view(src, ip, im, xt::all(), xt::all()),
            dst_view,
            xt::view(norm_pos, im, xt::all(), xt::all(), xt::all()),
            ignore_tile_edge
          );
        }
      }
#if defined(FOAM_WITH_TBB)
    }
  );
#endif
}

template<typename G>
template<typename M, typename E, EnableIf<std::decay_t<M>, IsModulesVector>, EnableIf<E, IsImageArray>>
void Detector1MGeometryBase<G>::positionAllModules(M&& src, E& dst, bool ignore_tile_edge) const
{
  auto ms = src[0].shape();
  auto ss = std::array<int, 4> {static_cast<int>(ms[0]),
                                static_cast<int>(src.size()),
                                static_cast<int>(ms[1]),
                                static_cast<int>(ms[2])};
  auto ds = dst.shape();
  this->checkShape(ss, ds);

  int n_pulses = ss[0];
  auto norm_pos = static_cast<const G*>(this)->corner_pos_ / static_cast<const G*>(this)->pixelSize();
#if defined(FOAM_WITH_TBB)
  tbb::parallel_for(tbb::blocked_range2d<int>(0, n_modules, 0, n_pulses),
    [&src, &dst, &norm_pos, ignore_tile_edge, this] (const tbb::blocked_range2d<int> &block)
    {
      for(int im=block.rows().begin(); im != block.rows().end(); ++im)
      {
        for(int ip=block.cols().begin(); ip != block.cols().end(); ++ip)
        {
#else
      for (int im = 0; im < n_modules; ++im)
      {
        for (int ip = 0; ip < n_pulses; ++ip)
        {
#endif
          auto&& dst_view = xt::view(dst, ip, xt::all(), xt::all());
          positionModule(
            xt::view(src[im], ip, xt::all(), xt::all()),
            dst_view,
            xt::view(norm_pos, im, xt::all(), xt::all(), xt::all()),
            ignore_tile_edge
          );
        }
      }
#if defined(FOAM_WITH_TBB)
    }
  );
#endif
}

template<typename G>
std::pair<typename Detector1MGeometryBase<G>::vector2dType,
          typename Detector1MGeometryBase<G>::vector2dType>
Detector1MGeometryBase<G>::assembledDim() const
{
  auto min_xyz = xt::amin(static_cast<const G*>(this)->corner_pos_,
                          {0, 1, 2}) / static_cast<const G*>(this)->pixelSize();
  auto max_xyz = xt::amax(static_cast<const G*>(this)->corner_pos_,
                          {0, 1, 2}) / static_cast<const G*>(this)->pixelSize();

  return {
    vector2dType { max_xyz[1] - min_xyz[1], max_xyz[0] - min_xyz[0]},
    vector2dType {             -min_xyz[0],             -min_xyz[1]}
  };
}

template<typename G>
template<typename SrcShape, typename DstShape>
void Detector1MGeometryBase<G>::checkShape(const SrcShape& ss, const DstShape& ds) const
{
  if (ss[0] != ds[0])
  {
    std::stringstream fmt;
    fmt << "Modules data and output array have different memory cells: "
        << ss[0] << " and " << ds[0] << "!";
    throw std::invalid_argument(fmt.str());
  }

  if (ss[1] != G::n_modules)
  {
    std::stringstream fmt;
    fmt << "Expected " << G::n_modules << " modules, get " << ss[1] << "!";
    throw std::invalid_argument(fmt.str());
  }

  if (ss[2] != G::module_shape[0] || ss[3] != G::module_shape[1])
  {
    std::stringstream fmt;
    fmt << "Expected modules with shape (" << G::module_shape[0] << ", " << G::module_shape[1]
        << ") modules, get (" << ss[2] << ", " << ss[3] << ")!";
    throw std::invalid_argument(fmt.str());
  }

  auto as = assembledShape();
  if (as[0] < 1024 or as[1] < 1024 or as[0] > 1536 or as[1] > 1536)
  {
    std::stringstream fmt;
    fmt << "Expected output array with shape (" << as[0] << ", " << as[1]
        << "). Side length of a 1M detector must be within [1024, 1536]!";
    throw std::invalid_argument(fmt.str());
  }

  if (as[0] != ds[1] | as[1] != ds[2])
  {
    std::stringstream fmt;
    fmt << "Expected output array with shape (" << as[0] << ", " << as[1]
        << ") modules, get (" << ds[1] << ", " << ds[2] << ")!";
    throw std::invalid_argument(fmt.str());
  }
}

template<typename G>
template<typename M, typename N, typename T>
void Detector1MGeometryBase<G>::positionModule(M&& src, N& dst, T&& pos, bool ignore_tile_edge) const
{
  static_cast<const G*>(this)->positionModuleImp(std::forward<M>(src), dst, std::forward<T>(pos), ignore_tile_edge);
}

/**
 * AGIPD-1M geometry
 *
 *
 * Layout of AGIPD-1M:              Tile layout for each module:
 * (looking along the beam)
 *
 *     Q4M1    |    Q1M1            Q1 and Q2: T8 T7 T6 T5 T4 T3 T2 T1
 *     Q4M2    |    Q1M2            Q3 and q4: T1 T2 T3 T4 T5 T6 T7 T8
 *     Q4M3    |    Q1M3
 *     Q4M4    |    Q1M4
 *  -----------------------
 *     Q3M1    |    Q2M1
 *     Q3M2    |    Q2M2
 *     Q3M3    |    Q2M3
 *     Q3M4    |    Q2M4
 *
 * The quadrant positions refer to the first pixel
 * (top-right corners for Q1, Q2 and bottom-left corners for Q3, Q4)
 * of the first module in each quadrant.
 *
 * For details, please see
 * https://extra-geom.readthedocs.io/en/latest/geometry.html#agipd-1m
 *
 */
class AGIPD_1MGeometry : public Detector1MGeometryBase<AGIPD_1MGeometry>
{
public:

  static const shapeType module_shape;
  static const shapeType tile_shape;
  static const int n_tiles_per_module = 8; // number of tiles per module
  static const quadOrientType quad_orientations;
private:

  xt::xtensor_fixed<double, xt::xshape<n_modules, n_tiles_per_module, 2, 3>> corner_pos_;

  friend Detector1MGeometryBase<AGIPD_1MGeometry>;

  template<typename M, typename N, typename T>
  void positionModuleImp(M&& src, N& dst, T&& pos, bool ignore_tile_edge) const;

public:

  static const vectorType& pixelSize()
  {
    static const vectorType pixel_size {2e-4, 2e-4, 1.};
    return pixel_size;
  }

  AGIPD_1MGeometry();

  explicit
  AGIPD_1MGeometry(const std::array<std::array<std::array<double, 3>, n_tiles_per_module>, n_modules>& positions);

  ~AGIPD_1MGeometry() = default;
};

// (ss/x, fs/y) This should be called 'data_shape' instead of 'module_shape'
const AGIPD_1MGeometry::shapeType AGIPD_1MGeometry::module_shape {512, 128};
// (fs/y, ss/x)
const AGIPD_1MGeometry::shapeType AGIPD_1MGeometry::tile_shape {128, 64};
constexpr int AGIPD_1MGeometry::n_tiles_per_module;
const AGIPD_1MGeometry::quadOrientType AGIPD_1MGeometry::quad_orientations {
  std::array<int, 2>{1, -1},
  std::array<int, 2>{1, -1},
  std::array<int, 2>{-1, 1},
  std::array<int, 2>{-1, 1}
};

AGIPD_1MGeometry::AGIPD_1MGeometry()
{
  // first pixel position of each module
  // (upper-right for Q1 and Q2, lower-left for Q3 and Q4) positions
  xt::xtensor_fixed<double, xt::xshape<n_modules, 3>> m_pos {
    { -512,  512, 0},
    { -512,  384, 0},
    { -512,  256, 0},
    { -512,  128, 0},
    { -512,    0, 0},
    { -512, -128, 0},
    { -512, -256, 0},
    { -512, -384, 0},
    {  512, -128, 0},
    {  512, -256, 0},
    {  512, -384, 0},
    {  512, -512, 0},
    {  512,  384, 0},
    {  512,  256, 0},
    {  512,  128, 0},
    {  512,    0, 0}
  };

  xt::xtensor_fixed<double, xt::xshape<n_modules, n_tiles_per_module, 3>> positions;
  auto ht = static_cast<double>(tile_shape[0]);
  auto wt = static_cast<double>(tile_shape[1]);
  for (int im = 0; im < n_modules; ++im)
  {
    auto orient = quad_orientations[im / 4];
    for (int it = 0; it < n_tiles_per_module; ++it)
    {
      positions(im, it, 0) = m_pos(im, 0) + orient[0] * it * wt;
      positions(im, it, 1) = m_pos(im, 1);
      positions(im, it, 2) = m_pos(im, 2);
    }
  }

  positions *= pixelSize();

  for (int im = 0; im < n_modules; ++im)
  {
    auto orient = quad_orientations[im / 4];
    for (int it = 0; it < n_tiles_per_module; ++it)
    {
      for (int j = 0; j < 3; ++j) corner_pos_(im, it, 0, j) = positions(im, it, j);
      // calculate the position of the diagonal corner
      corner_pos_(im, it, 1, 0) = positions(im, it, 0) + orient[0] * wt * pixelSize()(0);
      corner_pos_(im, it, 1, 1) = positions(im, it, 1) + orient[1] * ht * pixelSize()(1);
      corner_pos_(im, it, 1, 2) = 0.0;
    }
  }
}

AGIPD_1MGeometry::AGIPD_1MGeometry(
  const std::array<std::array<std::array<double, 3>, n_tiles_per_module>, n_modules>& positions)
{
  auto ht = static_cast<double>(tile_shape[0]);
  auto wt = static_cast<double>(tile_shape[1]);
  for (int im = 0; im < n_modules; ++im)
  {
    auto orient = quad_orientations[im / 4];

    for (int it = 0; it < n_tiles_per_module; ++it)
    {
      for (int j = 0; j < 3; ++j) corner_pos_(im, it, 0, j) = positions[im][it][j];
      // calculate the position of the diagonal corner
      corner_pos_(im, it, 1, 0) = positions[im][it][0] + orient[0] * wt * pixelSize()(0);
      corner_pos_(im, it, 1, 1) = positions[im][it][1] + orient[1] * ht * pixelSize()(1);
      corner_pos_(im, it, 1, 2) = 0.0;
    }
  }
}

template<typename M, typename N, typename T>
void AGIPD_1MGeometry::positionModuleImp(M&& src, N& dst, T&& pos, bool ignore_tile_edge) const
{
  auto center = assembledDim().second;
  auto shape = src.shape(); // caveat: shape has layout (y, x)
  int n_tiles = n_tiles_per_module;
  int wt = tile_shape[1];
  int ht = tile_shape[0];

  int edge = 0;
  if (ignore_tile_edge) edge = 1;

  for (int it = 0; it < n_tiles; ++it)
  {
    auto x0 = pos(it, 0, 0);
    auto y0 = pos(it, 0, 1);

    int ix_dir = (pos(it, 1, 0) - x0 > 0) ? 1 : -1;
    int iy_dir = (pos(it, 1, 1) - y0 > 0) ? 1 : -1;

    int ix0 = it * wt;
    int iy0 = 0;

    int ix0_dst = ix_dir > 0 ? std::floor(x0 + center[0]) : std::ceil(x0 + center[0]) - 1;
    int iy0_dst = iy_dir > 0 ? std::floor(y0 + center[1]) : std::ceil(y0 + center[1]) - 1;
    for (int iy = iy0 + edge, iy_dst = iy0_dst + iy_dir * edge; iy < iy0 + ht - edge; ++iy, iy_dst += iy_dir)
    {
      for (int ix = ix0 + edge, ix_dst = ix0_dst + ix_dir * edge; ix < ix0 + wt - edge; ++ix, ix_dst += ix_dir)
      {
        dst(iy_dst, ix_dst) = src(ix, iy); // (fs/y, ss/x)
      }
    }
  }
}

/**
 * LPD-1M geometry
 *
 *
 * Layout of LPD-1M:                   Tile layout for each module:
 * (looking along the beam)
 *
 *  Q4M4  Q4M1 | Q1M4  Q1M1            T16 T01
 *             |                       T15 T02
 *             |                       T14 T03
 *  Q4M3  Q4M2 | Q1M3  Q1M2            T13 T05
 *  -----------------------            T12 T05
 *  Q3M4  Q3M1 | Q2M4  Q2M1            T11 T06
 *             |                       T10 T07
 *             |                       T09 T08
 *  Q3M3  Q3M2 | Q2M3  Q2M2
 *
 * The quadrant positions refer to the corner of each quadrant where module 4,
 * tile 16 is positioned. This is the corner of the last pixel as the data is
 * stored. In the initial detector layout, the corner positions are for the
 * top left corner of the quadrant, looking along the beam.
 *
 * For details, please see
 * https://extra-geom.readthedocs.io/en/latest/geometry.html#lpd-1m
 *
 */
class LPD_1MGeometry : public Detector1MGeometryBase<LPD_1MGeometry>
{
public:

  static const shapeType module_shape;
  static const shapeType tile_shape;
  static const int n_tiles_per_module = 16; // number of tiles per module
  static const quadOrientType quad_orientations;
private:

  xt::xtensor_fixed<double, xt::xshape<n_modules, n_tiles_per_module, 2, 3>> corner_pos_;

  friend Detector1MGeometryBase<LPD_1MGeometry>;

  template<typename M, typename N, typename T>
  void positionModuleImp(M&& src, N& dst, T&& pos, bool ignore_tile_edge) const;

public:

  static const vectorType& pixelSize()
  {
    static const vectorType pixel_size {5e-4, 5e-4, 1.};
    return pixel_size;
  }

  LPD_1MGeometry();

  explicit
  LPD_1MGeometry(const std::array<std::array<std::array<double, 3>, n_tiles_per_module>, n_modules>& positions);

  ~LPD_1MGeometry() = default;
};

// (ss/y, fs/x)
const LPD_1MGeometry::shapeType LPD_1MGeometry::module_shape {256, 256};
// (ss/y, fs/x)
const LPD_1MGeometry::shapeType LPD_1MGeometry::tile_shape {32, 128};
constexpr int LPD_1MGeometry::n_tiles_per_module;
const LPD_1MGeometry::quadOrientType LPD_1MGeometry::quad_orientations {
  std::array<int, 2>{1, 1},
  std::array<int, 2>{1, 1},
  std::array<int, 2>{1, 1},
  std::array<int, 2>{1, 1}
};

LPD_1MGeometry::LPD_1MGeometry()
{
  // last pixel position (upper-left corner) of each module
  xt::xtensor_fixed<double, xt::xshape<n_modules, 3>> m_pos {
    { -256,  512, 0},
    { -256,  256, 0},
    {    0,  256, 0},
    {    0,  512, 0},
    { -256,    0, 0},
    { -256, -256, 0},
    {    0, -256, 0},
    {    0,    0, 0},
    {  256,    0, 0},
    {  256, -256, 0},
    {  512, -256, 0},
    {  512,    0, 0},
    {  256,  512, 0},
    {  256,  256, 0},
    {  512,  256, 0},
    {  512,  512, 0}
  };

  xt::xtensor_fixed<double, xt::xshape<n_modules, n_tiles_per_module, 3>> positions;
  auto ht = static_cast<double>(tile_shape[0]);
  auto wt = static_cast<double>(tile_shape[1]);
  for (int im = 0; im < n_modules; ++im)
  {
    for (int it = 0; it < n_tiles_per_module; ++it)
    {
      positions(im, it, 0) = m_pos(im, 0) - (1 - it / 8) * wt;
      positions(im, it, 1) = m_pos(im, 1) - (it < 8 ? (it % 8) * ht : (7 - it % 8) * ht);
      positions(im, it, 2) = m_pos(im, 2);
    }
  }

  // last pixel -> first pixel
  positions -= xt::xtensor_fixed<double, xt::xshape<3>>({wt, ht, 0});
  positions *= pixelSize();

  for (int im = 0; im < n_modules; ++im)
  {
    for (int it = 0; it < n_tiles_per_module; ++it)
    {
      for (int j = 0; j < 3; ++j) corner_pos_(im, it, 0, j) = positions(im, it, j);
      // calculate the position of the diagonal corner
      corner_pos_(im, it, 1, 0) = positions(im, it, 0) + wt * pixelSize()(0);
      corner_pos_(im, it, 1, 1) = positions(im, it, 1) + ht * pixelSize()(1);
      corner_pos_(im, it, 1, 2) = 0.0;
    }
  }
}

LPD_1MGeometry::LPD_1MGeometry(
  const std::array<std::array<std::array<double, 3>, n_tiles_per_module>, n_modules>& positions)
{
  for (int im = 0; im < n_modules; ++im)
  {
    for (int it = 0; it < n_tiles_per_module; ++it)
    {
      for (int j = 0; j < 3; ++j) corner_pos_(im, it, 0, j) = positions[im][it][j];
      // calculate the position of the diagonal corner
      corner_pos_(im, it, 1, 0) = positions[im][it][0] + static_cast<double>(tile_shape[1]) * pixelSize()(0);
      corner_pos_(im, it, 1, 1) = positions[im][it][1] + static_cast<double>(tile_shape[0]) * pixelSize()(1);
      corner_pos_(im, it, 1, 2) = 0.0;
    }
  }
}

template<typename M, typename N, typename T>
void LPD_1MGeometry::positionModuleImp(M&& src, N& dst, T&& pos, bool ignore_tile_edge) const
{
  auto center = assembledDim().second;
  auto shape = src.shape(); // caveat: shape has layout (y, x)
  int n_tiles = n_tiles_per_module;
  int wt = tile_shape[1];
  int ht = tile_shape[0];

  int edge = 0;
  if (ignore_tile_edge) edge = 1;

  for (int it = 0; it < n_tiles; ++it)
  {
    auto x0 = pos(it, 0, 0);
    auto y0 = pos(it, 0, 1);

    int ix_dir = (pos(it, 1, 0) - x0 > 0) ? 1 : -1;
    int iy_dir = (pos(it, 1, 1) - y0 > 0) ? 1 : -1;

    int ix0 = (it / 8) * wt;
    int iy0 = it < 8 ? (7 - it % 8) * ht : (it % 8) * ht;
    int ix0_dst = ix_dir > 0 ? std::floor(x0 + center[0]) : std::ceil(x0 + center[0]) - 1;
    int iy0_dst = iy_dir > 0 ? std::floor(y0 + center[1]) : std::ceil(y0 + center[1]) - 1;
    for (int iy = iy0 + edge, iy_dst = iy0_dst + iy_dir * edge; iy < iy0 + ht - edge; ++iy, iy_dst += iy_dir)
    {
      for (int ix = ix0 + edge, ix_dst = ix0_dst + ix_dir * edge; ix < ix0 + wt - edge; ++ix, ix_dst += ix_dir)
      {
        dst(iy_dst, ix_dst) = src(iy, ix);
      }
    }
  }
}

/**
 * DSSC-1M geometry
 *
 *
 * Layout of DSSC-1M:
 * (looking along the beam)
 *
 *      Q4M1    |    Q1M4
 *      Q4M2    |    Q1M3
 *      Q4M3    |    Q1M2
 *      Q4M4    |    Q1M1
 *  -------------------------
 *      Q3M1    |    Q2M4
 *      Q3M2    |    Q2M3
 *      Q3M3    |    Q2M2
 *      Q3M4    |    Q2M1
 *
 * The quadrant positions refer to the bottom-right corner of each quadrant,
 * looking along the beam.
 *
 */
class DSSC_1MGeometry : public Detector1MGeometryBase<DSSC_1MGeometry>
{
public:

  static const shapeType module_shape;
  static const shapeType tile_shape;
  static const int n_tiles_per_module = 2; // number of tiles per module
  static const quadOrientType quad_orientations;

private:

  xt::xtensor_fixed<double, xt::xshape<n_modules, n_tiles_per_module, 2, 3>> corner_pos_;

  friend Detector1MGeometryBase<DSSC_1MGeometry>;

  template<typename M, typename N, typename T>
  void positionModuleImp(M&& src, N& dst, T&& pos, bool ignore_tile_edge) const;

public:

  static const vectorType& pixelSize()
  {
    // Hexagonal pixels:
    //   Measuring in terms of the step within a row, the
    // step to the next row of hexagons is 1.5/sqrt(3).
    static const double step_size = 236e-6;
    // fs/x, ss/y
    static const vectorType pixel_size {step_size, step_size * 1.5 / sqrt(3.), 1.};
    return pixel_size;
  }

  DSSC_1MGeometry();

  explicit
  DSSC_1MGeometry(const std::array<std::array<std::array<double, 3>, n_tiles_per_module>, n_modules>& positions);
  // TODO: another constructor with geometry file and quadrant positions

  ~DSSC_1MGeometry() = default;

};

// (ss/y, fs/x)
const DSSC_1MGeometry::shapeType DSSC_1MGeometry::module_shape {128, 512};
// (ss/y, fs/x)
const DSSC_1MGeometry::shapeType DSSC_1MGeometry::tile_shape {128, 256};
constexpr int DSSC_1MGeometry::n_tiles_per_module;
const DSSC_1MGeometry::quadOrientType DSSC_1MGeometry::quad_orientations {
  std::array<int, 2>{-1, 1},
  std::array<int, 2>{-1, 1},
  std::array<int, 2>{1, -1},
  std::array<int, 2>{1, -1}
};

DSSC_1MGeometry::DSSC_1MGeometry()
{
  // first pixel position of each module
  // (lower-left for Q1 and Q2, upper-right for Q3 and Q4) positions
  xt::xtensor_fixed<double, xt::xshape<n_modules, n_tiles_per_module, 3>> positions {
    {{   0,    0, 0}, {-256,    0, 0}},
    {{   0,  128, 0}, {-256,  128, 0}},
    {{   0,  256, 0}, {-256,  256, 0}},
    {{   0,  384, 0}, {-256,  384, 0}},
    {{   0, -512, 0}, {-256, -512, 0}},
    {{   0, -384, 0}, {-256, -384, 0}},
    {{   0, -256, 0}, {-256, -256, 0}},
    {{   0, -128, 0}, {-256, -128, 0}},
    {{   0,    0, 0}, { 256,    0, 0}},
    {{   0, -128, 0}, { 256, -128, 0}},
    {{   0, -256, 0}, { 256, -256, 0}},
    {{   0, -384, 0}, { 256, -384, 0}},
    {{   0,  512, 0}, { 256,  512, 0}},
    {{   0,  384, 0}, { 256,  384, 0}},
    {{   0,  256, 0}, { 256,  256, 0}},
    {{   0,  128, 0}, { 256,  128, 0}},
  };
  positions *= pixelSize();

  for (int im = 0; im < n_modules; ++im)
  {
    auto orient = quad_orientations[im / 4];

    for (int it = 0; it < n_tiles_per_module; ++it)
    {
      for (int j = 0; j < 3; ++j) corner_pos_(im, it, 0, j) = positions(im, it, j);
      // calculate the position of the diagonal corner
      corner_pos_(im, it, 1, 0) = positions(im, it, 0)
                                  + orient[0] * static_cast<double>(tile_shape[1]) * pixelSize()(0);
      corner_pos_(im, it, 1, 1) = positions(im, it, 1)
                                  + orient[1] * static_cast<double>(tile_shape[0]) * pixelSize()(1);
      corner_pos_(im, it, 1, 2) = 0.0;
    }
  }
}

DSSC_1MGeometry::DSSC_1MGeometry(
  const std::array<std::array<std::array<double, 3>, n_tiles_per_module>, n_modules>& positions)
{
  for (int im = 0; im < n_modules; ++im)
  {
    auto orient = quad_orientations[im / 4];

    for (int it = 0; it < n_tiles_per_module; ++it)
    {
      for (int j = 0; j < 3; ++j) corner_pos_(im, it, 0, j) = positions[im][it][j];
      // calculate the position of the diagonal corner
      corner_pos_(im, it, 1, 0) = positions[im][it][0]
                                  + orient[0] * static_cast<double>(tile_shape[1]) * pixelSize()(0);
      corner_pos_(im, it, 1, 1) = positions[im][it][1]
                                  + orient[1] * static_cast<double>(tile_shape[0]) * pixelSize()(1);
      corner_pos_(im, it, 1, 2) = 0.0;
    }
  }
}

template<typename M, typename N, typename T>
void DSSC_1MGeometry::positionModuleImp(M&& src, N& dst, T&& pos, bool ignore_tile_edge) const
{
  auto center = assembledDim().second;
  int n_tiles = n_tiles_per_module;
  int wt = tile_shape[1];
  int ht = tile_shape[0];

  int edge = 0;
  if (ignore_tile_edge) edge = 1;

  for (int it = 0; it < n_tiles; ++it)
  {
    auto x0 = pos(it, 0, 0);
    auto y0 = pos(it, 0, 1);

    int ix_dir = (pos(it, 1, 0) - x0 > 0) ? 1 : -1;
    int iy_dir = (pos(it, 1, 1) - y0 > 0) ? 1 : -1;

    int ix0 = it * wt;
    int iy0 = 0;
    int ix0_dst = ix_dir > 0 ? std::floor(x0 + center[0]) : std::ceil(x0 + center[0]) - 1;
    int iy0_dst = iy_dir > 0 ? std::floor(y0 + center[1]) : std::ceil(y0 + center[1]) - 1;
    for (int iy = iy0 + edge, iy_dst = iy0_dst + iy_dir * edge; iy < ht - edge; ++iy, iy_dst += iy_dir)
    {
      for (int ix = ix0 + edge, ix_dst = ix0_dst + ix_dir * edge; ix < ix0 + wt - edge; ++ix, ix_dst += ix_dir)
      {
        dst(iy_dst, ix_dst) = src(iy, ix);
      }
    }
  }
}

}; //foam


#endif //EXTRA_FOAM_DETECTOR_GEOMETRY_H
