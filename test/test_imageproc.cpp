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

#include "f_imageproc.hpp"

namespace foam
{
namespace test
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

TEST(TestNanmeanImageArray, TestTwoImages)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);
  auto inf = std::numeric_limits<float>::infinity();

  // lvalue
  xt::xtensor<float, 2> img1 {{1.f, -inf, 2.f}, {4.f, 5.f, nan}};
  xt::xtensor<float, 2> img2 {{1.f, 2.f, 3.f}, {inf, nan, 6.f}};
  xt::xtensor<float, 2> ret_gt {{1.f, -inf, 2.5}, {inf, 5.f, 6.f}};
  EXPECT_THAT(nanmeanImageArray(img1, img2), ElementsAreArray(ret_gt));

  xt::xtensor<float, 2> img3 {{1.f, 2.f, 3.f}, {inf, nan, nan}};
  EXPECT_THAT(nanmeanImageArray(img1, img3), ElementsAre(1.f, -inf, 2.5, inf, 5.f, nan_mt));

  // rvalue
  EXPECT_THAT(nanmeanImageArray(std::move(img1), std::move(img2)), ElementsAreArray(ret_gt));
}

TEST(TestMaskImageData, Test2DRaw)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  xt::xtensor<float, 2> img {{1.f, nan, 3.f}, {4.f, 5.f, nan}};
  xt::xtensor<float, 2> img_gt {{1.f, 0.f, 3.f}, {4.f, 5.f, 0.f}};
  maskImageData(img);
  EXPECT_THAT(img, ElementsAreArray(img_gt));
  EXPECT_THAT(img(0, 0), NanSensitiveFloatEq(1.f));
}

TEST(TestMaskImageData, Test2DThresholdMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};

  xt::xtensor<float, 2> masked_img_gt {};
  maskImageData(img, 2, 4);
  EXPECT_THAT(img, ElementsAre(nan_mt, 2.f, 3.f, 4.f, nan_mt, nan_mt));
}

TEST(TestMaskImageData, Test2DImageMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};

  xt::xtensor<bool, 2> mask_w {{true, true}, {true, true}};
  EXPECT_THROW(maskImageData(img, mask_w), std::invalid_argument);

  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};
  maskImageData(img, mask);
  EXPECT_THAT(img, ElementsAre(nan_mt, nan_mt, 3.f, nan_mt, 5.f, 6.f));
}

TEST(TestMaskImageData, Test2DBothMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};

  xt::xtensor<bool, 2> mask_w {{true, true}, {true, true}};
  EXPECT_THROW(maskImageData(img, mask_w), std::invalid_argument);

  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};
  maskImageData(img, mask, 2, 4);
  EXPECT_THAT(img, ElementsAre(nan_mt, nan_mt, 3.f, nan_mt, nan_mt, nan_mt));
}

TEST(TestMaskImageData, Test3DRaw)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();

  xt::xtensor<float, 3> imgs {{{1.f, 2.f, nan}, {4.f, nan, 6.f}},
                              {{nan, 2.f, 3.f}, {4.f, nan, 6.f}}};
  xt::xtensor<float, 3> imgs_gt {{{1.f, 2.f, 0.f}, {4.f, 0.f, 6.f}},
                                 {{0.f, 2.f, 3.f}, {4.f, 0.f, 6.f}}};
  maskImageData(imgs);
  EXPECT_THAT(imgs, ElementsAreArray(imgs_gt));
  EXPECT_THAT(imgs(0, 0, 0), NanSensitiveFloatEq(1.f));
}

TEST(TestMaskImageData, Test3DThresholdMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs {{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}},
                              {{2.f, 3.f, 4.f}, {5.f, 6.f, 7.f}}};

  maskImageData(imgs, 2, 4);
  EXPECT_THAT(xt::view(imgs, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, 2.f, 3.f, 4.f, nan_mt, nan_mt));
  EXPECT_THAT(xt::view(imgs, 1, xt::all(), xt::all()),
              ElementsAre(2.f, 3.f, 4.f, nan_mt, nan_mt, nan_mt));
}

TEST(TestMaskImageData, Test3DImageMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs {{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}},
                              {{2.f, 3.f, 4.f}, {5.f, 6.f, 7.f}}};

  xt::xtensor<bool, 2> mask_w {{true, true}, {true, true}};
  EXPECT_THROW(maskImageData(imgs, mask_w), std::invalid_argument);

  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};
  maskImageData(imgs, mask);
  EXPECT_THAT(xt::view(imgs, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, nan_mt, 3.f, nan_mt, 5.f, 6.f));
  EXPECT_THAT(xt::view(imgs, 1, xt::all(), xt::all()),
              ElementsAre(nan_mt, nan_mt, 4.f, nan_mt, 6.f, 7.f));
}

TEST(TestMaskImageData, Test3DBothMask)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = 0.f; // NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs {{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}},
                              {{2.f, 3.f, 4.f}, {5.f, 6.f, 7.f}}};

  xt::xtensor<bool, 2> mask_w {{true, true}, {true, true}};
  EXPECT_THROW(maskImageData(imgs, mask_w, 2, 4), std::invalid_argument);

  xt::xtensor<bool, 2> mask {{true, true, false}, {true, false, false}};
  maskImageData(imgs, mask, 2, 4);
  EXPECT_THAT(xt::view(imgs, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, nan_mt, 3.f, nan_mt, nan_mt, nan_mt));
  EXPECT_THAT(xt::view(imgs, 1, xt::all(), xt::all()),
              ElementsAre(nan_mt, nan_mt, 4.f, nan_mt, nan_mt, nan_mt));
}

TEST(TestMovingAvgImageData, Test2D)
{
  xt::xtensor<float, 2> img1 {{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}};
  xt::xtensor<float, 2> img2 {{2.f, 3.f, 4.f}, {4.f, 5.f, 6.f}};
  xt::xtensor<float, 2> img3 {{2.f, 3.f}, {4.f, 5.f}};
  EXPECT_THROW(movingAvgImageData(img1, img2, 0), std::invalid_argument);
  EXPECT_THROW(movingAvgImageData(img1, img3, 2), std::invalid_argument);

  movingAvgImageData(img1, img2, 2);
  xt::xtensor<float, 2> ma {{1.5f, 2.5f, 3.5f}, {3.5f, 4.5f, 5.5f}};
  EXPECT_THAT(img1, ElementsAreArray(ma));
  EXPECT_THAT(img1(0, 0), NanSensitiveFloatEq(1.5f));
}

TEST(TestMovingAvgImageData, Test2DWithNan)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img1 {{nan, 2.f, nan}, {3.f, 4.f, 5.f}};
  xt::xtensor<float, 2> img2 {{2.f, 3.f, nan}, {4.f, 5.f, 6.f}};

  movingAvgImageData(img1, img2, 2);
  EXPECT_THAT(img1, ElementsAre(nan_mt, 2.5f, nan_mt, 3.5f, 4.5f, 5.5f));
}

TEST(TestMovingAvgImageData, Test3D)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();

  xt::xtensor<float, 3> imgs1 {{{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}},
                               {{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}}};
  xt::xtensor<float, 3> imgs2 {{{2.f, 3.f, 4.f}, {4.f, 5.f, 6.f}},
                               {{2.f, 3.f, 4.f}, {4.f, 5.f, 6.f}}};
  xt::xtensor<float, 3> imgs3 {{{2.f, 3.f}, {4.f, 5.f}}, {{2.f, 3.f}, {4.f, 5.f}}};
  EXPECT_THROW(movingAvgImageData(imgs1, imgs2, 0), std::invalid_argument);
  EXPECT_THROW(movingAvgImageData(imgs1, imgs3, 2), std::invalid_argument);

  movingAvgImageData(imgs1, imgs2, 2);
  EXPECT_THAT(xt::view(imgs1, 0, xt::all(), xt::all()),
              ElementsAre(1.5f, 2.5f, 3.5f, 3.5f, 4.5f, 5.5f));
  EXPECT_THAT(xt::view(imgs1, 1, xt::all(), xt::all()),
              ElementsAre(1.5f, 2.5f, 3.5f, 3.5f, 4.5f, 5.5f));
  EXPECT_THAT(imgs1(0, 0, 0), NanSensitiveFloatEq(1.5f));
}

TEST(TestMovingAvgImageData, Test3DWithNaN)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs1 {{{nan, 2.f, nan}, {3.f, 4.f, 5.f}},
                               {{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}}};
  xt::xtensor<float, 3> imgs2 {{{2.f, 3.f, nan}, {4.f, 5.f, 6.f}},
                               {{2.f, nan, 4.f}, {4.f, nan, 6.f}}};

  movingAvgImageData(imgs1, imgs2, 2);
  xt::xtensor<float, 3> ma {{{nan, 2.5f, nan}, {3.5f, 4.5f, 5.5f}},
                            {{1.5f, nan, 3.5f}, {3.5f, nan, 5.5f}}};

  EXPECT_THAT(xt::view(imgs1, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, 2.5f, nan_mt, 3.5f, 4.5f, 5.5f));
  EXPECT_THAT(xt::view(imgs1, 1, xt::all(), xt::all()),
              ElementsAre(1.5f, nan_mt, 3.5f, 3.5f, nan_mt, 5.5f));
}

TEST(correctImageData, TestOffset3D)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs {{{nan, 2.f, nan}, {3.f, 4.f, 5.f}},
                              {{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}}};
  xt::xtensor<float, 3> offset {{{2.f, 4.f, nan}, {4.f, 5.f, 6.f}},
                                {{1.f, nan, 2.f}, {4.f, nan, 6.f}}};
  correctImageData<OffsetPolicy>(imgs, offset);

  EXPECT_THAT(xt::view(imgs, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, -2.f, nan_mt, -1.f, -1.f, -1.f));
  EXPECT_THAT(xt::view(imgs, 1, xt::all(), xt::all()),
              ElementsAre(0.f, nan_mt, 1.f, -1.f, nan_mt, -1.f));
}

TEST(correctImageData, TestOffset2D)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{nan, 2.f, nan}, {3.f, 4.f, 5.f}};
  xt::xtensor<float, 2> offset {{2.f, 4.f, nan}, {4.f, 4.f, 6.f}};
  correctImageData<OffsetPolicy>(img, offset);

  EXPECT_THAT(img, ElementsAre(nan_mt, -2.f, nan_mt, -1.f, 0.f, -1.f));
}

TEST(correctImageData, TestGain3D)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs {{{nan, 2.f, nan}, {3.f, 4.f, 5.f}},
                              {{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}}};
  xt::xtensor<float, 3> gain {{{2.f, 4.f, nan}, {4.f, 5.f, 6.f}},
                              {{1.f, nan, 2.f}, {4.f, nan, 6.f}}};
  correctImageData<GainPolicy>(imgs, gain);

  EXPECT_THAT(xt::view(imgs, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, 8.f, nan_mt, 12.f, 20.f, 30.f));
  EXPECT_THAT(xt::view(imgs, 1, xt::all(), xt::all()),
              ElementsAre(1.f, nan_mt, 6.f, 12.f, nan_mt, 30.f));
}

TEST(correctImageData, TestGain2D)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{nan, 2.f, nan}, {3.f, 4.f, 5.f}};
  xt::xtensor<float, 2> offset {{2.f, 4.f, nan}, {4.f, 4.f, 6.f}};
  correctImageData<GainPolicy>(img, offset);

  EXPECT_THAT(img, ElementsAre(nan_mt, 8.f, nan_mt, 12.f, 16.f, 30.f));
}

TEST(correctImageData, TestGainOffset3D)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 3> imgs {{{nan, 2.f, nan}, {3.f, 4.f, 5.f}},
                              {{1.f, 2.f, 3.f}, {3.f, 4.f, 5.f}}};
  xt::xtensor<float, 3> offset {{{2.f, 4.f, nan}, {4.f, 5.f, 6.f}},
                                {{1.f, nan, 2.f}, {4.f, nan, 6.f}}};
  xt::xtensor<float, 3> gain {{{1.f, 2.f, 1.f}, {2.f, 1.f, 2.f}},
                              {{1.f, 1.f, 2.f}, {1.f, 2, 2.f}}};
  correctImageData(imgs, gain, offset);

  EXPECT_THAT(xt::view(imgs, 0, xt::all(), xt::all()),
              ElementsAre(nan_mt, -4.f, nan_mt, -2.f, -1.f, -2.f));
  EXPECT_THAT(xt::view(imgs, 1, xt::all(), xt::all()),
              ElementsAre(0.f, nan_mt, 2.f, -1.f, nan_mt, -2.f));
}

TEST(correctImageData, TestGainOffset2D)
{
  auto nan = std::numeric_limits<float>::quiet_NaN();
  auto nan_mt = NanSensitiveFloatEq(nan);

  xt::xtensor<float, 2> img {{nan, 2.f, nan}, {3.f, 4.f, 5.f}};
  xt::xtensor<float, 2> offset {{2.f, 4.f, nan}, {4.f, 4.f, 6.f}};
  xt::xtensor<float, 2> gain {{2.f, 1.f, 2.f}, {1.f, 2.f, 2.f}};
  correctImageData(img, gain, offset);

  EXPECT_THAT(img, ElementsAre(nan_mt, -2.f, nan_mt, -1.f, 0.f, -2.f));
}

} // test
} // foam
