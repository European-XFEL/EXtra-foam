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

#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include "f_canny.hpp"

namespace foam
{
namespace test
{

using ::testing::Each;
using ::testing::Eq;
using ::testing::ElementsAre;

static constexpr auto nan = std::numeric_limits<float>::quiet_NaN();

TEST(TestCanny, TestEdgeDetection)
{
  xt::xtensor<float, 2> src {xt::ones<float>({6, 8})};
  xt::xtensor<float, 2> dst {xt::zeros<float>({6, 8})};
  xt::view(src, xt::range(3, 4), xt::all()) = 0;

  cannyEdge(src, dst, 0, 2);
  EXPECT_THAT(xt::view(dst, 0, xt::all()), Each(Eq(0)));
  EXPECT_THAT(xt::view(dst, 1, xt::all()), Each(Eq(0)));
  EXPECT_THAT(xt::view(dst, 2, xt::all()), ElementsAre(0, 1, 1, 1, 1, 1, 1, 0));
  EXPECT_THAT(xt::view(dst, 3, xt::all()), Each(Eq(0)));
  EXPECT_THAT(xt::view(dst, 4, xt::all()), ElementsAre(0, 1, 1, 1, 1, 1, 1, 0));
  EXPECT_THAT(xt::view(dst, 5, xt::all()), Each(Eq(0)));

  src(2, 2) = nan;
  src(2, 3) = nan;
  cannyEdge(src, dst, 0, 2);
  EXPECT_THAT(xt::view(dst, 1, xt::all()), Each(Eq(0)));
  EXPECT_THAT(xt::view(dst, 2, xt::all()), ElementsAre(0, 0, 0, 0, 0, 1, 1, 0));
  EXPECT_THAT(xt::view(dst, 3, xt::all()), Each(Eq(0)));
  EXPECT_THAT(xt::view(dst, 4, xt::all()), ElementsAre(0, 1, 1, 1, 1, 1, 1, 0));

  // test threshold
  cannyEdge(src, dst, 0, 4);
  EXPECT_THAT(xt::view(dst, 2, xt::all()), ElementsAre(0, 0, 0, 0, 0, 1, 1, 0));
  cannyEdge(src, dst, 0, 4.01);
  EXPECT_THAT(xt::view(dst, 2, xt::all()), ElementsAre(0, 0, 0, 0, 0, 0, 0, 0));

  // test a different output type
  xt::xtensor<bool, 2> dst_bool {xt::zeros<bool>({6, 8})};
  cannyEdge(src, dst_bool, 0, 2);
  EXPECT_THAT(xt::view(dst, 1, xt::all()), Each(Eq(false)));
  EXPECT_THAT(xt::view(dst_bool, 2, xt::all()), ElementsAre(false, false, false, false, false, true, true, false));
  EXPECT_THAT(xt::view(dst, 3, xt::all()), Each(Eq(false)));
  EXPECT_THAT(xt::view(dst_bool, 4, xt::all()), ElementsAre(false, true, true, true, true, true, true, false));

}


} // test
} // foam