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

#include "xtensor/xtensor.hpp"
#include "xtensor-blas/xlinalg.hpp"

namespace foam
{
namespace test
{

using ::testing::ElementsAre;

TEST(TestMatrixEigenValues, TestGeneral)
{
  xt::xtensor<double, 2> a {
    {1, 0, 0},
    {1, 2, 0},
    {2, 3, 3}
  };

  auto ret = xt::linalg::eig(a);
  EXPECT_THAT(std::get<0>(ret), ElementsAre(3, 2, 1));

  xt::xtensor<double, 2> eigen_vec_gt {
    {0, 0, 0.666667}, {0, 0.316228, -0.666667}, {1, -0.948683, 0.333333}
  };
  EXPECT_TRUE(xt::allclose(std::get<1>(ret), eigen_vec_gt));
}

TEST(TestInv, TestMultiplicative)
{
  xt::xtensor<double, 2> a {{3.0, 3.5}, {3.2, 3.6}};
  auto ret = xt::linalg::inv(a);

  xt::xtensor<double, 2> inv_gt {
    {-9.0, 8.75}, {8.0, -7.5}
  };
  EXPECT_TRUE(xt::allclose(ret, inv_gt));
}

TEST(TestLstsq, TestGeneral)
{
  xt::xtensor<double, 2> mat {{0, 1}, {1, 1}, {2, 1}, {3, 1}};
  xt::xtensor<double, 1> y {-1.0, 0.2, 0.9, 2.1};

  auto ret = xt::linalg::lstsq(mat, y);
  xt::xtensor<double, 1> lstsq_gt {1., -0.95};
  EXPECT_TRUE(xt::allclose(std::get<0>(ret), lstsq_gt));
}

}
}

