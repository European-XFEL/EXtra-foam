/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#ifndef LIVE_VIEW_CONFIG_H
#define LIVE_VIEW_CONFIG_H

#include <xtensor/xarray.hpp>

#include <tbb/concurrent_queue.h>

using ImageQueue = tbb::concurrent_bounded_queue<xt::xarray<float>>;

#endif //LIVE_VIEW_CONFIG_H
