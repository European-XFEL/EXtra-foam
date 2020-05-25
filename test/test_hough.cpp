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

#include "f_hough.hpp"


namespace foam
{
namespace test
{

TEST(TestConcentricCircle, TestGeneral)
{
  xt::xtensor<float, 2> src {xt::ones<float>({6, 8})};

  concentricCircle(src, 0, 6, 0, 8, 7);

}

} // test
} // foam