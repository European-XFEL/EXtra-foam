/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#include <memory>

#include "pybind11/pybind11.h"
#include "xtensor/xarray.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"


/**
 * Moving average of a scalar data.
 */
template<typename T>
class MovingAverage {

  T data_; // moving average
  size_t window_;
  size_t count_;

public:
  explicit MovingAverage(T v)
    : data_(v), count_(1), window_(1) {
  };

  ~MovingAverage() = default;

  void set(T v) {
    if (window_ > 1 && count_ <= window_) {
      if (count_ < window_) {
        ++count_;
        data_ = data_ + (v - data_) / count_;
      } else { // count_ == window_
        data_ = data_ + (v - data_) / window_; // this is an approximation
      }
    } else {
      data_ = v;
      count_ = 1;
    }
  }

  T get() { return data_; }

  void setWindow(size_t v) {
    if (! v) throw std::invalid_argument("Moving average window must be positive!");

    window_ = v;
  }

  size_t window() const { return window_; }

  size_t count() const { return count_; }
};


/**
 * Moving average of a numpy array.
 */
template<typename T>
class MovingAverageArray {

protected:
  xt::pyarray<T> data_; // moving average
  size_t window_;
  size_t count_;

public:
  explicit MovingAverageArray(const xt::pyarray<T>& arr)
    : data_(arr), window_(1), count_(1) {
  };

  virtual ~MovingAverageArray() = default;

  void set(const xt::pyarray<T>& arr) {
    if (window_ > 1 && count_ <= window_ && arr.shape() == data_.shape()) {
      if (count_ < window_) {
        ++count_;
        data_ = data_ + (arr - data_) / count_;
      } else {
        // this is an approximation
        data_ = data_ + (arr - data_) / window_;
      }
    } else {
      data_ = arr;
      count_ = 1;
    }
  };

  // TODO: fix it
  xt::pyarray<T>& get() { return data_; }

  void setWindow(size_t v) {
    if (! v) throw std::invalid_argument("Moving average window must be positive!");

    window_ = v;
  }

  size_t window() const { return window_; }

  size_t count() const { return count_; }
};


template <typename T>
class RawImageData : public MovingAverageArray<T> {

public:
  explicit RawImageData(const xt::pyarray<T>& arr) : MovingAverageArray<T>(arr) {
  };

  ~RawImageData() final = default;

  size_t nImages() const {
    if (pulseResolved()) return this->data_.shape()[0];
    return 1;
  }

  bool pulseResolved() const { return this->data_.shape().size() == 3; }
};


namespace py = pybind11;

template<typename T>
void declare_MovingAverage(py::module &m, const std::string &type_str) {
  using Class = MovingAverage<T>;

  std::string py_class_name = std::string("MovingAverage") + type_str;
  py::class_<Class>(m, py_class_name.c_str())
    .def(py::init<T>())
    .def("get", &Class::get)
    .def("set", &Class::set)
    .def("window", &Class::window)
    .def("setWindow", &Class::setWindow)
    .def("count", &Class::count);
}


template<typename T>
void declare_MovingAverageArray(py::module &m, const std::string &type_str) {
  using Class = MovingAverageArray<T>;

  std::string py_class_name = std::string("MovingAverageArray") + type_str;
  py::class_<Class>(m, py_class_name.c_str())
    .def(py::init<const xt::pyarray<T>&>())
    .def("get", &Class::get)
    .def("set", &Class::set)
    .def("window", &Class::window)
    .def("setWindow", &Class::setWindow)
    .def("count", &Class::count);
}


template<typename T>
void declare_RawImageData(py::module &m, const std::string &type_str) {
  using Class = RawImageData<T>;

  std::string py_class_name = std::string("RawImageData") + type_str;
  py::class_<Class, MovingAverageArray<T>>(m, py_class_name.c_str())
    .def(py::init<const xt::pyarray<T>&>())
    .def("nImages", &RawImageData<T>::nImages)
    .def("pulseResolved", &RawImageData<T>::pulseResolved);
}


PYBIND11_MODULE(datamodel, m) {
  xt::import_numpy();

  declare_MovingAverage<float>(m, "Float");
  declare_MovingAverage<double>(m, "Double");

  declare_MovingAverageArray<float>(m, "Float");
  declare_MovingAverageArray<double>(m, "Double");

  declare_RawImageData<float>(m, "Float");
  declare_RawImageData<double>(m, "Double");
}
