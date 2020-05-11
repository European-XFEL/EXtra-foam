/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <memory>

#include "xtensor/xio.hpp"

#include "f_geometry_1m.hpp"

namespace foam
{
namespace test
{

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::NanSensitiveFloatEq;
using ::testing::FloatEq;

template<typename T>
class Geometry1M : public ::testing::Test
{

protected:
  Geometry1M() : geom_(std::make_unique<T>()) {}

  ~Geometry1M() = default;

  std::unique_ptr<T> geom_;

  float nan = std::numeric_limits<float>::quiet_NaN();
  DSSC_1MGeometry::shapeType shape = geom_->assembledShape();
  DSSC_1MGeometry::shapeType center = geom_->assembledCenter();

  int np_ = 2; // number of pulses
  int mw_ = T::module_shape[1];
  int mh_ = T::module_shape[0];
  int nm_ = T::n_modules;
  int tw_ = T::tile_shape[1];
  int th_ = T::tile_shape[0];
};


using Geometry1MTypes = ::testing::Types<DSSC_1MGeometry, LPD_1MGeometry, AGIPD_1MGeometry>;
TYPED_TEST_CASE(Geometry1M, Geometry1MTypes);


TYPED_TEST(Geometry1M, testAssembledShapeAndCenter)
{
  EXPECT_THAT(this->shape, ElementsAre(1024, 1024));
  EXPECT_THAT(this->center, ElementsAre(512, 512));
}

TYPED_TEST(Geometry1M, testAssemblingShapeCheck)
{
  // src and dst have different memory cells
  xt::xtensor<float, 4> src1 {xt::ones<float>({this->np_ - 1, this->nm_, this->mh_, this->mw_})};
  xt::xtensor<float, 3> dst1 {xt::empty<float>({this->np_, this->shape[0], this->shape[1]})};
  EXPECT_THROW(this->geom_->positionAllModules(src1, dst1), std::invalid_argument);

  // src has incorrect shape
  xt::xtensor<float, 3> dst2{xt::empty<float>({this->np_, this->shape[0], this->shape[1]})};
  xt::xtensor<float, 4> src2_1{xt::ones<float>({this->np_, this->nm_ - 1, this->mh_, this->mw_})};
  EXPECT_THROW(this->geom_->positionAllModules(src2_1, dst2), std::invalid_argument);
  xt::xtensor<float, 4> src2_2 { xt::ones<float>({this->np_, this->nm_, this->mh_+1, this->mw_}) };
  EXPECT_THROW(this->geom_->positionAllModules(src2_2, dst2), std::invalid_argument);
  xt::xtensor<float, 4> src2_3 { xt::ones<float>({this->np_, this->nm_, this->mh_, this->mw_+1}) };
  EXPECT_THROW(this->geom_->positionAllModules(src2_3, dst2), std::invalid_argument);

  // dst has incorrect shape
  xt::xtensor<float, 4> src3 { xt::ones<float>({this->np_, this->nm_, this->mh_, this->mw_}) };
  xt::xtensor<float, 3> dst3 {
      xt::empty<float>({this->np_, static_cast<int>(this->shape[0]) + 1, static_cast<int>(this->shape[1])}) };
  EXPECT_THROW(this->geom_->positionAllModules(src3, dst3), std::invalid_argument);
}

TYPED_TEST(Geometry1M, testPositionAllModulesSingle)
{
  xt::xtensor<float, 3> src { xt::ones<float>({this->nm_, this->mh_, this->mw_}) };
  xt::xtensor<float, 2> dst {
    xt::empty<float>({static_cast<int>(this->shape[0]), static_cast<int>(this->shape[1])}) };

  this->geom_->positionAllModules(src, dst); // test no throw
  EXPECT_THAT(dst, ::testing::Each(1.f));
}

TYPED_TEST(Geometry1M, testPositionAllModulesArray)
{
  xt::xtensor<float, 4> src { xt::ones<float>({this->np_, this->nm_, this->mh_, this->mw_}) };
  xt::xtensor<float, 3> dst {
      xt::empty<float>({this->np_, static_cast<int>(this->shape[0]), static_cast<int>(this->shape[1])}) };

  this->geom_->positionAllModules(src, dst); // test no throw
  EXPECT_THAT(dst, ::testing::Each(1.f));
}

TYPED_TEST(Geometry1M, testPositionAllModulesVector)
{
  std::vector<xt::xtensor<float, 3>> src;
  for (auto i = 0; i < this->nm_; ++i) src.emplace_back(xt::ones<float>({this->np_, this->mh_, this->mw_}));
  xt::xtensor<float, 3> dst {
      xt::empty<float>({this->np_, static_cast<int>(this->shape[0]), static_cast<int>(this->shape[1])}) };

  this->geom_->positionAllModules(src, dst); // test no throw
  EXPECT_THAT(dst, ::testing::Each(1.f));
}

TYPED_TEST(Geometry1M, testIgnoreTileEdge)
{
  xt::xtensor<float, 3> dst {
    xt::empty<float>({this->np_, static_cast<int>(this->shape[0]), static_cast<int>(this->shape[1])}) };
  dst.fill(this->nan);

  xt::xtensor<float, 4> src { xt::ones<float>({this->np_, this->nm_, this->mh_, this->mw_}) };
  this->geom_->positionAllModules(src, dst, true);

  EXPECT_THAT(xt::view(dst, 0, 0, xt::all()), ::testing::Each(NanSensitiveFloatEq(this->nan)));

  for (int bottom=0, top=this->th_ - 1; bottom < this->shape[0]; top += this->th_, bottom += this->th_)
  {
    EXPECT_THAT(xt::view(dst, xt::all(), bottom, xt::all()), ::testing::Each(NanSensitiveFloatEq(this->nan)));
    EXPECT_THAT(xt::view(dst, xt::all(), top, xt::all()), ::testing::Each(NanSensitiveFloatEq(this->nan)));
  }

  for (int left = 0, right = this->tw_ - 1; right < this->shape[1]; left += this->tw_, right += this->tw_)
  {
    EXPECT_THAT(xt::view(dst, xt::all(), xt::all(), left), ::testing::Each(NanSensitiveFloatEq(this->nan)));
    EXPECT_THAT(xt::view(dst, xt::all(), xt::all(), right), ::testing::Each(NanSensitiveFloatEq(this->nan)));
  }
}

TYPED_TEST(Geometry1M, testDismanleShapeCheck)
{
  // src and dst have different memory cells
  xt::xtensor<float, 3> src1 {
    xt::ones<float>({this->np_ - 1, static_cast<int>(this->shape[0]), static_cast<int>(this->shape[1])}) };
  xt::xtensor<float, 4> dst1 { xt::empty<float>({this->np_, this->nm_, this->mh_, this->mw_}) };
  EXPECT_THROW(this->geom_->dismantleAllModules(src1, dst1), std::invalid_argument);

  // src has incorrect shape
  xt::xtensor<float, 4> dst2 { xt::empty<float>({this->np_, this->nm_, this->mh_, this->mw_}) };
  xt::xtensor<float, 3> src2_1 {
    xt::ones<float>({this->np_, static_cast<int>(this->shape[0]) + 1, static_cast<int>(this->shape[1])}) };
  EXPECT_THROW(this->geom_->dismantleAllModules(src2_1, dst2), std::invalid_argument);
  xt::xtensor<float, 3> src2_2 {
    xt::ones<float>({this->np_, static_cast<int>(this->shape[0]), static_cast<int>(this->shape[1]) + 1}) };
  EXPECT_THROW(this->geom_->dismantleAllModules(src2_2, dst2), std::invalid_argument);

  // dst has incorrect shape
  xt::xtensor<float, 3> src3 {
    xt::ones<float>({this->np_, static_cast<int>(this->shape[0]), static_cast<int>(this->shape[1])}) };
  xt::xtensor<float, 4> dst3_1 { xt::empty<float>({this->np_, this->nm_ + 1, this->mh_, this->mw_}) };
  EXPECT_THROW(this->geom_->dismantleAllModules(src3, dst3_1), std::invalid_argument);
  xt::xtensor<float, 4> dst3_2 { xt::empty<float>({this->np_, this->nm_, this->mh_ + 1, this->mw_}) };
  EXPECT_THROW(this->geom_->dismantleAllModules(src3, dst3_2), std::invalid_argument);
  xt::xtensor<float, 4> dst3_3 { xt::empty<float>({this->np_, this->nm_, this->mh_, this->mw_ + 1}) };
  EXPECT_THROW(this->geom_->dismantleAllModules(src3, dst3_3), std::invalid_argument);
}

TYPED_TEST(Geometry1M, testDismentalAllModulesSingle)
{
  xt::xtensor<float, 3> src { xt::ones<float>({this->nm_, this->mh_, this->mw_}) };
  xt::xtensor<float, 2> dst {
    xt::zeros<float>({static_cast<int>(this->shape[0]), static_cast<int>(this->shape[1])}) };
  xt::xtensor<float, 3> dst_src { xt::zeros<float>(src.shape()) };

  this->geom_->positionAllModules(src, dst);
  this->geom_->dismantleAllModules(dst, dst_src); // test no throw
  EXPECT_THAT(dst_src, ::testing::Each(1.f));
}

TYPED_TEST(Geometry1M, testDismentalAllModules)
{
  xt::xtensor<float, 4> src { xt::ones<float>({this->np_, this->nm_, this->mh_, this->mw_}) };
  xt::xtensor<float, 3> dst {
    xt::zeros<float>({this->np_, static_cast<int>(this->shape[0]), static_cast<int>(this->shape[1])}) };
  xt::xtensor<float, 4> dst_src { xt::zeros<float>(src.shape()) };

  this->geom_->positionAllModules(src, dst);
  this->geom_->dismantleAllModules(dst, dst_src); // test no throw
  EXPECT_THAT(dst_src, ::testing::Each(1.f));
}

} //test
} //foam