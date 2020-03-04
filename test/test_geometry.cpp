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

#include "f_geometry.hpp"

namespace foam
{
namespace test
{

template<typename T>
class Test1MGeometry : public ::testing::Test
{

protected:
  Test1MGeometry() : geom_(std::make_unique<T>()) {}

  ~Test1MGeometry() = default;

  std::unique_ptr<T> geom_;

  int mw_ = T::module_shape[1];
  int mh_ = T::module_shape[0];
  int nm_ = T::n_modules;
};



using Geometry1MTypes = ::testing::Types<DSSC_1MGeometry, LPD_1MGeometry, AGIPD_1MGeometry>;
TYPED_TEST_CASE(Test1MGeometry, Geometry1MTypes);


TYPED_TEST(Test1MGeometry, testMemoryCellCheck)
{
  xt::xtensor<float, 3> dst{xt::empty<float>({4, 1024, 1024})};
  xt::xtensor<float, 4> src1{xt::zeros<float>({2, this->nm_, this->mh_, this->mw_})};
  EXPECT_THROW(this->geom_->positionAllModules(src1, dst), std::invalid_argument);
}

TYPED_TEST(Test1MGeometry, testModuleNumberCheck)
{
  xt::xtensor<float, 3> dst{xt::empty<float>({2, 1024, 1024})};
  xt::xtensor<float, 4> src{xt::zeros<float>({2, this->nm_ / 2, this->mh_, this->mw_})};
  EXPECT_THROW(this->geom_->positionAllModules(src, dst), std::invalid_argument);
}

TYPED_TEST(Test1MGeometry, testModuleShapeCheck)
{
  xt::xtensor<float, 3> dst{xt::empty<float>({4, 1024, 1024})};

  xt::xtensor<float, 4> src1 { xt::zeros<float>({2, this->nm_, this->mh_+1, this->mw_}) };
  EXPECT_THROW(this->geom_->positionAllModules(src1, dst), std::invalid_argument);
  xt::xtensor<float, 4> src2 { xt::zeros<float>({2, this->nm_, this->mh_, this->mw_+1}) };
  EXPECT_THROW(this->geom_->positionAllModules(src2, dst), std::invalid_argument);
}

TYPED_TEST(Test1MGeometry, testAssembledShapeCheck)
{
  auto shape = this->geom_->assembledShape();
  xt::xtensor<float, 3> dst{xt::empty<float>({2, static_cast<int>(shape[0]) + 1, static_cast<int>(shape[1])})};
  xt::xtensor<float, 4> modules { xt::zeros<float>({2, this->nm_, this->mh_, this->mw_}) };
  EXPECT_THROW(this->geom_->positionAllModules(modules, dst), std::invalid_argument);
}

TYPED_TEST(Test1MGeometry, testPositionAllModulesOnline)
{
  auto shape = this->geom_->assembledShape();
  xt::xtensor<float, 3> dst{xt::empty<float>({2, static_cast<int>(shape[0]), static_cast<int>(shape[1])})};
  xt::xtensor<float, 4> modules { xt::zeros<float>({2, this->nm_, this->mh_, this->mw_}) };

  this->geom_->positionAllModules(modules, dst); // test no throw
}

TYPED_TEST(Test1MGeometry, testPositionAllModulesFile)
{
  auto shape = this->geom_->assembledShape();
  xt::xtensor<float, 3> dst{xt::empty<float>({2, static_cast<int>(shape[0]), static_cast<int>(shape[1])})};
  std::vector<xt::xtensor<float, 3>> modules;
  for (auto i = 0; i < this->nm_; ++i) modules.emplace_back(xt::zeros<float>({2, this->mh_, this->mw_}));

  this->geom_->positionAllModules(modules, dst);
}

} //test
} //foam