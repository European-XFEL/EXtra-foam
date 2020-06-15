/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#include <chrono>

#include <extra-foam/f_imageproc.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>


int main()
{
  using namespace std::chrono;
  using namespace xt::placeholders;

  xt::xtensor<float, 3> arr = xt::random::rand<float>({64, 1200, 1024});
  // TODO:: shall we have a nan type
  xt::view(arr, xt::all(), xt::range(_, _, 2), xt::range(_, _, 4)) = std::numeric_limits<float>::quiet_NaN();

  xt::xtensor<bool, 2> image_mask = xt::ones<bool>({1200, 1024});
  auto start = std::chrono::high_resolution_clock::now();
  foam::maskImageDataNan<xt::xtensor<float, 3>, xt::xtensor<bool, 2>, float>(arr, image_mask, -1, 1);
  std::cout << "- Mask of a train with 64 pulses takes: "
            << (duration_cast<milliseconds>(high_resolution_clock::now() - start)).count()
            << " ms\n";

  start = std::chrono::high_resolution_clock::now();
  auto avg = foam::nanmeanImageArray(arr);
  std::cout << "- Average images of a train with 64 pulses takes: "
            << (duration_cast<milliseconds>(high_resolution_clock::now() - start)).count()
            << " ms\n";

  std::cout << std::endl;
  return 0;
}
