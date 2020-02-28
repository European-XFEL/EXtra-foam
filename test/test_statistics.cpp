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
#include "xtensor/xtensor.hpp"

#include "f_statistics.hpp"

namespace foam
{
namespace test
{

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::NanSensitiveFloatEq;
using ::testing::FloatEq;

TEST(TestNanmean, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{1.f, -1.f, 1.f}, {4.f, 5.f, nan}};
  EXPECT_EQ(2.f, foam::nanmean(img));
}

TEST(TestNansum, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{1.f, -1.f, 1.f}, {4.f, 5.f, nan}};
  EXPECT_EQ(10.f, foam::nansum(img));
}

} //test
} //foam