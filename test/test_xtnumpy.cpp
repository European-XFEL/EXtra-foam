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

TEST(TestMaskImage, TestThresholdMask)
{
  // threshold mask
  xt::xtensor<float, 2> img {{1, 2, 3}, {4, 5, 6}};

  xt::xtensor<float, 2> masked_img_gt {{0, 2, 3}, {4, 0, 0}};
  maskImage(img, 2, 4);
  EXPECT_EQ(img, masked_img_gt);
}

TEST(TestMaskImage, TestImageMask)
{
  // threshold mask
  xt::xtensor<float, 2> img {{1, 2, 3}, {4, 5, 6}};
  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};

  xt::xtensor<float, 2> masked_img_gt {{0, 0, 3}, {0, 5, 6}};
  maskImage(img, mask);
  EXPECT_EQ(img, masked_img_gt);
}

TEST(TestMaskTrainImages, TestThresholdMask)
{
  xt::xtensor<float, 3> imgs {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};

  xt::xtensor<float, 3> masked_imgs_gt {{{0, 2, 3}, {4, 0, 0}}, {{0, 2, 3}, {4, 0, 0}}};
  maskTrainImages(imgs, 2, 4);
  EXPECT_EQ(imgs, masked_imgs_gt);
}

TEST(TestXtMaskTrainImages, TestThresholdMask)
{
  xt::xtensor<float, 3> imgs {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};

  xt::xtensor<float, 3> masked_imgs_gt {{{0, 2, 3}, {4, 0, 0}}, {{0, 2, 3}, {4, 0, 0}}};
  xtMaskTrainImages(imgs, 2, 4);
  EXPECT_EQ(imgs, masked_imgs_gt);
}

TEST(TestMaskTrainImages, TestImageMask)
{
  xt::xtensor<float, 3> imgs {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}};
  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};

  xt::xtensor<float, 3> masked_imgs_gt {{{0, 0, 3}, {0, 5, 6}}, {{0, 0, 3}, {0, 5, 6}}};
  maskTrainImages(imgs, mask);
  EXPECT_EQ(imgs, masked_imgs_gt);
}

TEST(TestNanToZeroImage, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  xt::xtensor<float, 2> img {{1, nan, 3}, {4, 5, nan}};
  xt::xtensor<float, 2> img_gt {{1, 0, 3}, {4, 5, 0}};
  nanToZeroImage(img);
  EXPECT_EQ(img, img_gt);
}

TEST(TestNanToZeroTrainImages, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  xt::xtensor<float, 3> imgs {{{1, 2, nan}, {4, nan, 6}}, {{nan, 2, 3}, {4, nan, 6}}};
  xt::xtensor<float, 3> imgs_gt {{{1, 2, 0}, {4, 0, 6}}, {{0, 2, 3}, {4, 0, 6}}};
  nanToZeroTrainImages(imgs);
  EXPECT_EQ(imgs, imgs_gt);
}

} // fai
