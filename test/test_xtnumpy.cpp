/**
 * Offline and online data analysis and visualization tool for azimuthal
 * integration of different data acquired with various detectors at
 * European XFEL.
 *
 * Unittest for xtnumpy.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#include <array>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "xtensor/xtensor.hpp"

#include "image_proc.hpp"

namespace fai
{
namespace testing
{

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(TestMaskPulse, TestThresholdMask)
{
  // threshold mask
  xt::xtensor<float, 2> img {{1, 2, 3}, {4, 5, 6}};

  xt::xtensor<float, 2> masked_img_gt {{0, 2, 3}, {4, 0, 0}};
  maskPulse(img, 2, 4);
  EXPECT_THAT(img, ElementsAreArray(masked_img_gt));
}

TEST(TestMaskPulse, TestImageMask)
{
  // threshold mask
  xt::xtensor<float, 2> img {{1, 2, 3}, {4, 5, 6}};
  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};

  xt::xtensor<bool, 2> mask_w {{true, true}, {true, true}};
  EXPECT_THROW(maskPulse(img, mask_w), std::invalid_argument);

  xt::xtensor<float, 2> masked_img_gt {{0, 0, 3}, {0, 5, 6}};
  maskPulse(img, mask);
  EXPECT_THAT(img, ElementsAreArray(masked_img_gt));
}

TEST(TestMaskTrain, TestThresholdMask)
{
  xt::xtensor<float, 3> imgs {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};

  xt::xtensor<float, 3> masked_imgs_gt {{{0, 2, 3}, {4, 0, 0}}, {{0, 2, 3}, {4, 0, 0}}};
  maskTrain(imgs, 2, 4);
  EXPECT_THAT(imgs, ElementsAreArray(masked_imgs_gt));
}

TEST(TestXtMaskTrain, TestThresholdMask)
{
  xt::xtensor<float, 3> imgs {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};

  xt::xtensor<float, 3> masked_imgs_gt {{{0, 2, 3}, {4, 0, 0}}, {{0, 2, 3}, {4, 0, 0}}};
  xtMaskTrain(imgs, 2, 4);
  EXPECT_THAT(imgs, ElementsAreArray(masked_imgs_gt));
}

TEST(TestMaskTrain, TestImageMask)
{
  xt::xtensor<float, 3> imgs {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};
  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};

  xt::xtensor<bool, 2> mask_w {{true, true}, {true, true}};
  EXPECT_THROW(maskTrain(imgs, mask_w), std::invalid_argument);

  xt::xtensor<float, 3> masked_imgs_gt {{{0, 0, 3}, {0, 5, 6}}, {{0, 0, 3}, {0, 5, 6}}};
  maskTrain(imgs, mask);
  EXPECT_THAT(imgs, ElementsAreArray(masked_imgs_gt));
}

TEST(TestNanToZeroPulse, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  xt::xtensor<float, 2> img {{1, nan, 3}, {4, 5, nan}};
  xt::xtensor<float, 2> img_gt {{1, 0, 3}, {4, 5, 0}};
  nanToZeroPulse(img);
  EXPECT_THAT(img, ElementsAreArray(img_gt));
}

TEST(TestNanToZeroTrain, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();

  xt::xtensor<float, 3> imgs {{{1, 2, nan}, {4, nan, 6}}, {{nan, 2, 3}, {4, nan, 6}}};
  xt::xtensor<float, 3> imgs_gt {{{1, 2, 0}, {4, 0, 6}}, {{0, 2, 3}, {4, 0, 6}}};
  nanToZeroTrain(imgs);
  EXPECT_THAT(imgs, ElementsAreArray(imgs_gt));
}

TEST(TestMovingAveragePulse, TestGeneral)
{
  xt::xtensor<float, 2> img1 {{1, 2, 3}, {3, 4, 5}};
  xt::xtensor<float, 2> img2 {{2, 3, 4}, {4, 5, 6}};
  xt::xtensor<float, 2> img3 {{2, 3}, {4, 5}};
  EXPECT_THROW(movingAveragePulse(img1, img2, 0), std::invalid_argument);
  EXPECT_THROW(movingAveragePulse(img1, img3, 2), std::invalid_argument);

  movingAveragePulse(img1, img2, 2);
  xt::xtensor<float, 2> ma {{1.5, 2.5, 3.5}, {3.5, 4.5, 5.5}};
  EXPECT_THAT(img1, ElementsAreArray(ma));
}

TEST(TestMovingAveragePulse, TestWithNaN)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();

  xt::xtensor<float, 2> img1 {{nan, 2, nan}, {3, 4, 5}};
  xt::xtensor<float, 2> img2 {{  2, 3, nan}, {4, 5, 6}};

  movingAveragePulse(img1, img2, 2);
  xt::xtensor<float, 2> ma {{nan, 2.5, nan}, {3.5, 4.5, 5.5}};

  EXPECT_TRUE(std::isnan(img1(0, 0)));
  EXPECT_TRUE(std::isnan(img1(0, 2)));
  EXPECT_EQ(ma(1, 1), img1(1, 1));
}

TEST(TestMovingAverageTrain, TestGeneral)
{
  xt::xtensor<float, 3> imgs1 {{{1, 2, 3}, {3, 4, 5}}, {{1, 2, 3}, {3, 4, 5}}};
  xt::xtensor<float, 3> imgs2 {{{2, 3, 4}, {4, 5, 6}}, {{2, 3, 4}, {4, 5, 6}}};
  xt::xtensor<float, 3> imgs3 {{{2, 3}, {4, 5}}, {{2, 3}, {4, 5}}};
  EXPECT_THROW(movingAverageTrain(imgs1, imgs2, 0), std::invalid_argument);
  EXPECT_THROW(movingAverageTrain(imgs1, imgs3, 2), std::invalid_argument);

  movingAverageTrain(imgs1, imgs2, 2);
  xt::xtensor<float, 3> ma {{{1.5, 2.5, 3.5}, {3.5, 4.5, 5.5}}, {{1.5, 2.5, 3.5}, {3.5, 4.5, 5.5}}};
  EXPECT_THAT(ma, ElementsAreArray(imgs1));
}

TEST(TestMovingAverageTrain, TestWithNaN)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();

  xt::xtensor<float, 3> imgs1 {{{nan, 2, nan}, {3, 4, 5}}, {{1, 2, 3}, {3, 4, 5}}};
  xt::xtensor<float, 3> imgs2 {{{2, 3, nan}, {4, 5, 6}}, {{2, nan, 4}, {4, nan, 6}}};

  movingAverageTrain(imgs1, imgs2, 2);
  xt::xtensor<float, 3> ma {{{nan, 2.5, nan}, {3.5, 4.5, 5.5}}, {{1.5, nan, 3.5}, {3.5, nan, 5.5}}};

  EXPECT_TRUE(std::isnan(imgs1(0, 0, 0)));
  EXPECT_TRUE(std::isnan(imgs1(0, 0, 2)));
  EXPECT_TRUE(std::isnan(imgs1(1, 0, 1)));
  EXPECT_EQ(ma(0, 0, 1), imgs1(0, 0, 1));
}

} // testing
} // fai
