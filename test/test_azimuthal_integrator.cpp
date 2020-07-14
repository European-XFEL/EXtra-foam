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

#include "f_azimuthal_integrator.hpp"

namespace foam
{
namespace test
{

using ::testing::Each;
using ::testing::Eq;
using ::testing::ElementsAre;

static constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

TEST(TestAzimuthalIntegrator, TestIntegrator1D)
{
  xt::xtensor<float, 2> src = xt::arange(1024).reshape({16, 128});

  double distance = 0.2;
  double pixel1 = 1e-4;
  double pixel2 = 2e-4;
  double poni1 = -6 * pixel1;
  double poni2 = 130 * pixel2;
  double wavelength = 1e-10;
  AzimuthalIntegrator itgt(distance, poni1, poni2, pixel1, pixel2, wavelength);

  auto ret0 = itgt.integrate1d(src, 0);
  auto ret1 = itgt.integrate1d(src, 1);
  EXPECT_EQ(ret0, ret1);

  auto ret10 = itgt.integrate1d(src, 10);
  auto ret10_cut = itgt.integrate1d(src, 10, src.size());
  EXPECT_EQ(ret10.first, ret10_cut.first);
  EXPECT_THAT(ret10_cut.second, Each(Eq(0.)));

  auto ret100 = itgt.integrate1d(src, 999);

  // data has a single value
  xt::xtensor<double, 2> src2 = xt::ones<double>({16, 128});
  xt::arange(1024).reshape({16, 128});

  // integral source value type
  xt::xtensor<uint16_t, 2> src_int = xt::arange(1024).reshape({16, 128});
  auto ret10_uint16 = itgt.integrate1d(src, 10);
  EXPECT_EQ(ret10.first, ret10_uint16.first);
  EXPECT_THAT(ret10.second, ret10_uint16.second);
}

TEST(TestConcentricRingsFinder, TestGeneral)
{
  xt::xtensor<double, 2> src = xt::ones<double>({16, 128});

  double pixel_x = 2e-4;
  double pixel_y = 1e-4;
  double cx = 128;
  double cy = -6;
  size_t min_count = 32;

  ConcentricRingsFinder finder(pixel_x, pixel_y);
  finder.search(src, cx, cy, min_count);
  finder.integrate(src, cx, cy, min_count);
}

} //test
} //foam

