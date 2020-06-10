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

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#if defined(FOAM_USE_TBB)

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/tick_count.h"

namespace foam
{
namespace test
{

static const std::size_t N = 21;

// Finds largest matching substrings.
void SerialSubStringFinder (const std::string &str,
                            std::vector<std::size_t> &max_array,
                            std::vector<std::size_t> &pos_array) {
  for (std::size_t i = 0; i < str.size(); ++i) {
    std::size_t max_size = 0, max_pos = 0;
    for (std::size_t j = 0; j < str.size(); ++j)
      if (j != i) {
        std::size_t limit = str.size()-(std::max)(i, j);
        for (std::size_t k = 0; k < limit; ++k) {
          if (str[i+k] != str[j+k])
            break;
          if (k > max_size) {
            max_size = k;
            max_pos = j;
          }
        }
      }
    max_array[i] = max_size;
    pos_array[i] = max_pos;
  }
}


// Finds largest matching substrings (parallel version).
class SubStringFinder {
  const char *str_;
  const std::size_t len_;
  std::size_t *max_array_;
  std::size_t *pos_array_;
public:
  void operator() (const tbb::blocked_range<std::size_t>& r) const {
    for (std::size_t i = r.begin(); i != r.end(); ++i) {
      std::size_t max_size = 0, max_pos = 0;
      for (std::size_t j = 0; j < len_; ++j) {
        if (j != i) {
          std::size_t limit = len_ - (std::max)(i, j);
          for (std::size_t k = 0; k < limit; ++k) {
            if (str_[i+k] != str_[j+k])
              break;
            if (k > max_size) {
              max_size = k;
              max_pos = j;
            }
          }
        }
      }
      max_array_[i] = max_size;
      pos_array_[i] = max_pos;
    }
  }
  // We do not use std::vector for compatibility with offload execution
  SubStringFinder(const char *s, const std::size_t s_len, std::size_t *m, std::size_t *p) :
    str_(s), len_(s_len), max_array_(m), pos_array_(p) {}
};


TEST(TestTBB, GeneralSubStringFinder) {
  using namespace tbb;

  std::string str[N] = {std::string("a"), std::string("b")};
  for (std::size_t i = 2; i < N; ++i) str[i] = str[i-1]+str[i-2];
  std::string &to_scan = str[N-1];
  const std::size_t num_elem = to_scan.size();

  std::vector<std::size_t> max1(num_elem);
  std::vector<std::size_t> pos1(num_elem);
  std::vector<std::size_t> max2(num_elem);
  std::vector<std::size_t> pos2(num_elem);

  tick_count serial_t0 = tick_count::now();
  SerialSubStringFinder(to_scan, max2, pos2);
  std::cout << "Serial version 'SubStringFinder' ran in " << (tick_count::now() - serial_t0).seconds()
            << " seconds" << std::endl;

  tick_count parallel_t0 = tick_count::now();
  parallel_for(blocked_range<std::size_t>(0, num_elem, 100),
               SubStringFinder(to_scan.c_str(), num_elem, &max1[0], &pos1[0]));
  std::cout << "Parallel version 'SubStringFinder' ran in " <<  (tick_count::now() - parallel_t0).seconds()
            << " seconds" << std::endl;

  for (std::size_t i = 0; i < num_elem; ++i) {
    ASSERT_EQ(max1[i], max2[i]);
    ASSERT_EQ(pos1[i], pos2[i]);
  }
}

} // test
} // foam

#endif // FOAM_USE_TBB
