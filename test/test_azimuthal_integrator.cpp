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

TEST(TestAzimuthalIntegrate, TestGeneral)
{
  xt::xtensor<double, 2> src = xt::ones<double>({6, 8});

  double distance = 1.;
  double pixel1 = 1e-4;
  double pixel2 = 2e-4;
  double poni1 = 2 * pixel1;
  double poni2 = 4 * pixel2;
  double wavelength = 1e-10;
  size_t npt = 10;

  AzimuthalIntegrator itgt(distance, poni1, poni2, pixel1, pixel2, wavelength);
  auto ret = itgt.integrate1d(src, npt);
  std::cout << ret.first << "\n" << ret.second;
}


} //test
} //foam

