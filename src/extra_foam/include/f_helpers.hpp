/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#ifndef EXTRA_FOAM_F_HELPERS_H
#define EXTRA_FOAM_F_HELPERS_H

#include <array>

namespace foam {

/**
 * Calculate the intersection area of two rectangles.

 * @param rect1: (x1, y1, w1, h1) of rectangle 1.
 * @param rect2: (x2, y2, w2, h2) of rectangle 2.
 *
 * Note: (x, y) is the coordinate of the closest corner to the origin.
 *
 * @returns: (x, y, w, h) of the intersection area.
 */
inline std::array<int, 4>
intersection(const std::array<int, 4> &rect1, const std::array<int, 4> &rect2) {
  int x = std::max(rect1[0], rect2[0]);
  int xx = std::min(rect1[0] + rect1[2], rect2[0] + rect2[2]);
  int y = std::max(rect1[1], rect2[1]);
  int yy = std::min(rect1[1] + rect1[3], rect2[1] + rect2[3]);

  int w = xx - x;
  int h = yy - y;

  return {x, y, w, h};
}

}

#endif //EXTRA_FOAM_F_HELPERS_H
