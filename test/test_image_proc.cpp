/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
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

namespace foam
{
namespace testing
{

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::NanSensitiveFloatEq;
using ::testing::FloatEq;


TEST(TestNanmeanImageArray, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);
  auto inf = std::numeric_limits<float>::infinity();

  // lvalue
  xt::xtensor<float, 3> imgs {{{1.f, -inf, 2.f}, {4.f, 5.f, nan}},
                              {{1.f, 2.f, 3.f}, {inf, nan, 6.f}}};
  xt::xtensor<float, 2> ret_gt {{1.f, -inf, 2.5f}, {inf, 5.f, 6.f}};
  EXPECT_THAT(nanmeanImageArray(imgs), ElementsAreArray(ret_gt));

  imgs(0, 1, 1) = nan;
  EXPECT_THAT(nanmeanImageArray(imgs), ElementsAre(1.f, -inf, 2.5f, inf, nan_mt, 6.f));

  // rvalue
  EXPECT_THAT(nanmeanImageArray(std::move(imgs)), ElementsAre(1.f, -inf, 2.5f, inf, nan_mt, 6.f));
}

TEST(TestNanmeanTwoImages, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);
  auto inf = std::numeric_limits<float>::infinity();

  // lvalue
  xt::xtensor<float, 2> img1 {{1.f, -inf, 2.f}, {4.f, 5.f, nan}};
  xt::xtensor<float, 2> img2 {{1.f, 2.f, 3.f}, {inf, nan, 6.f}};
  xt::xtensor<float, 2> ret_gt {{1.f, -inf, 2.5}, {inf, 5.f, 6.f}};
  EXPECT_THAT(nanmeanTwoImages(img1, img2), ElementsAreArray(ret_gt));

  xt::xtensor<float, 2> img3 {{1.f, 2.f, 3.f}, {inf, nan, nan}};
  EXPECT_THAT(nanmeanTwoImages(img1, img3), ElementsAre(1.f, -inf, 2.5, inf, 5.f, nan_mt));

  // rvalue
  EXPECT_THAT(nanmeanTwoImages(std::move(img1), std::move(img2)), ElementsAreArray(ret_gt));
}

TEST(TestMaskImage, TestThresholdMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};

  xt::xtensor<float, 2> masked_img_gt {};
  maskImage(img, 2, 4);
  EXPECT_THAT(img, ElementsAre(nan_mt, 2.f, 3.f, 4.f, nan_mt, nan_mt));
}

TEST(TestMaskImage, TestImageMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};

  xt::xtensor<bool, 2> mask_w {{true, true}, {true, true}};
  EXPECT_THROW(maskImage(img, mask_w), std::invalid_argument);

  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};
  maskImage(img, mask);
  EXPECT_THAT(img, ElementsAre(nan_mt, nan_mt, 3.f, nan_mt, 5.f, 6.f));
}

TEST(TestMaskImage, TestBothMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};

  xt::xtensor<bool, 2> mask_w {{true, true}, {true, true}};
  EXPECT_THROW(maskImage(img, mask_w), std::invalid_argument);

  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};
  maskImage(img, mask, 2, 4);
  EXPECT_THAT(img, ElementsAre(nan_mt, nan_mt, 3.f, nan_mt, nan_mt, nan_mt));
}

TEST(TestMaskImageArray, TestThresholdMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs {{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}},
                              {{2.f, 3.f, 4.f}, {5.f, 6.f, 7.f}}};

  maskImageArray(imgs, 2, 4);
  EXPECT_THAT(xt::view(imgs, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, 2.f, 3.f, 4.f, nan_mt, nan_mt));
  EXPECT_THAT(xt::view(imgs, 1, xt::all(), xt::all()),
              ElementsAre(2.f, 3.f, 4.f, nan_mt, nan_mt, nan_mt));
}

TEST(TestMaskImageArray, TestImageMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs {{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}},
                              {{2.f, 3.f, 4.f}, {5.f, 6.f, 7.f}}};

  xt::xtensor<bool, 2> mask_w {{true, true}, {true, true}};
  EXPECT_THROW(maskImageArray(imgs, mask_w), std::invalid_argument);

  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};
  maskImageArray(imgs, mask);
  EXPECT_THAT(xt::view(imgs, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, nan_mt, 3.f, nan_mt, 5.f, 6.f));
  EXPECT_THAT(xt::view(imgs, 1, xt::all(), xt::all()),
              ElementsAre(nan_mt, nan_mt, 4.f, nan_mt, 6.f, 7.f));
}

TEST(TestMaskImageArray, TestBothMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs {{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}},
                              {{2.f, 3.f, 4.f}, {5.f, 6.f, 7.f}}};

  xt::xtensor<bool, 2> mask_w {{true, true}, {true, true}};
  EXPECT_THROW(maskImageArray(imgs, mask_w, 2, 4), std::invalid_argument);

  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};
  maskImageArray(imgs, mask, 2, 4);
  EXPECT_THAT(xt::view(imgs, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, nan_mt, 3.f, nan_mt, nan_mt, nan_mt));
  EXPECT_THAT(xt::view(imgs, 1, xt::all(), xt::all()),
              ElementsAre(nan_mt, nan_mt, 4.f, nan_mt, nan_mt, nan_mt));
}

TEST(TestNanToZeroImage, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  xt::xtensor<float, 2> img {{1.f, nan, 3.f}, {4.f, 5.f, nan}};
  xt::xtensor<float, 2> img_gt {{1.f, 0.f, 3.f}, {4.f, 5.f, 0.f}};
  nanToZeroImage(img);
  EXPECT_THAT(img, ElementsAreArray(img_gt));
  EXPECT_THAT(img(0, 0), NanSensitiveFloatEq(1.f));
}

TEST(TestNanToZeroImageArray, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();

  xt::xtensor<float, 3> imgs {{{1.f, 2.f, nan}, {4.f, nan, 6.f}},
                              {{nan, 2.f, 3.f}, {4.f, nan, 6.f}}};
  xt::xtensor<float, 3> imgs_gt {{{1.f, 2.f, 0.f}, {4.f, 0.f, 6.f}},
                                 {{0.f, 2.f, 3.f}, {4.f, 0.f, 6.f}}};
  nanToZeroImageArray(imgs);
  EXPECT_THAT(imgs, ElementsAreArray(imgs_gt));
  EXPECT_THAT(imgs(0, 0, 0), NanSensitiveFloatEq(1.f));
}

TEST(TestMovingAverageImage, TestGeneral)
{
  xt::xtensor<float, 2> img1 {{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}};
  xt::xtensor<float, 2> img2 {{2.f, 3.f, 4.f}, {4.f, 5.f, 6.f}};
  xt::xtensor<float, 2> img3 {{2.f, 3.f}, {4.f, 5.f}};
  EXPECT_THROW(movingAverageImage(img1, img2, 0), std::invalid_argument);
  EXPECT_THROW(movingAverageImage(img1, img3, 2), std::invalid_argument);

  movingAverageImage(img1, img2, 2);
  xt::xtensor<float, 2> ma {{1.5f, 2.5f, 3.5f}, {3.5f, 4.5f, 5.5f}};
  EXPECT_THAT(img1, ElementsAreArray(ma));
  EXPECT_THAT(img1(0, 0), NanSensitiveFloatEq(1.5f));
}

TEST(TestMovingAverageImage, TestWithNaN)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img1 {{nan, 2.f, nan}, {3.f, 4.f, 5.f}};
  xt::xtensor<float, 2> img2 {{2.f, 3.f, nan}, {4.f, 5.f, 6.f}};

  movingAverageImage(img1, img2, 2);
  EXPECT_THAT(img1, ElementsAre(nan_mt, 2.5f, nan_mt, 3.5f, 4.5f, 5.5f));
}

TEST(TestMovingAverageImageArray, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();

  xt::xtensor<float, 3> imgs1 {{{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}},
                               {{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}}};
  xt::xtensor<float, 3> imgs2 {{{2.f, 3.f, 4.f}, {4.f, 5.f, 6.f}},
                               {{2.f, 3.f, 4.f}, {4.f, 5.f, 6.f}}};
  xt::xtensor<float, 3> imgs3 {{{2.f, 3.f}, {4.f, 5.f}}, {{2.f, 3.f}, {4.f, 5.f}}};
  EXPECT_THROW(movingAverageImageArray(imgs1, imgs2, 0), std::invalid_argument);
  EXPECT_THROW(movingAverageImageArray(imgs1, imgs3, 2), std::invalid_argument);

  movingAverageImageArray(imgs1, imgs2, 2);
  EXPECT_THAT(xt::view(imgs1, 0, xt::all(), xt::all()),
              ElementsAre(1.5f, 2.5f, 3.5f, 3.5f, 4.5f, 5.5f));
  EXPECT_THAT(xt::view(imgs1, 1, xt::all(), xt::all()),
              ElementsAre(1.5f, 2.5f, 3.5f, 3.5f, 4.5f, 5.5f));
  EXPECT_THAT(imgs1(0, 0, 0), NanSensitiveFloatEq(1.5f));
}

TEST(TestMovingAverageImageArray, TestWithNaN)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs1 {{{nan, 2.f, nan}, {3.f, 4.f, 5.f}},
                               {{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}}};
  xt::xtensor<float, 3> imgs2 {{{2.f, 3.f, nan}, {4.f, 5.f, 6.f}},
                               {{2.f, nan, 4.f}, {4.f, nan, 6.f}}};

  movingAverageImageArray(imgs1, imgs2, 2);
  xt::xtensor<float, 3> ma {{{nan, 2.5f, nan}, {3.5f, 4.5f, 5.5f}},
                            {{1.5f, nan, 3.5f}, {3.5f, nan, 5.5f}}};

  EXPECT_THAT(xt::view(imgs1, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, 2.5f, nan_mt, 3.5f, 4.5f, 5.5f));
  EXPECT_THAT(xt::view(imgs1, 1, xt::all(), xt::all()),
              ElementsAre(1.5f, nan_mt, 3.5f, 3.5f, nan_mt, 5.5f));
}

TEST(TestSubtractDarkFromImageArray, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs {{{nan, 2.f, nan}, {3.f, 4.f, 5.f}},
                               {{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}}};
  xt::xtensor<float, 3> darks {{{2.f, 4.f, nan}, {4.f, 5.f, 6.f}},
                               {{1.f, nan, 2.f}, {4.f, nan, 6.f}}};
  subDarkImageArray(imgs, darks);

  EXPECT_THAT(xt::view(imgs, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, -2.f, nan_mt, -1.f, -1.f, -1.f));
  EXPECT_THAT(xt::view(imgs, 1, xt::all(), xt::all()),
              ElementsAre(0.f, nan_mt, 1.f, -1.f, nan_mt, -1.f));
}

TEST(TestSubtractDarkFromImage, TestGeneral)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{nan, 2.f, nan}, {3.f, 4.f, 5.f}};
  xt::xtensor<float, 2> dark {{2.f, 4.f, nan}, {4.f, 4.f, 6.f}};
  subDarkImage(img, dark);

  EXPECT_THAT(img, ElementsAre(nan_mt, -2.f, nan_mt, -1.f, 0.f, -1.f));
}

} // testing
} // foam
