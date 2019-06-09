/**
 * Offline and online data analysis and visualization tool for azimuthal
 * integration of different data acquired with various detectors at
 * European XFEL.
 *
 * ImageData.
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
 * Construct assembled detector image from modules data. For train-resolved
 * detectors, it simply copy? the original data.
 */

template<typename T>
class RawImageData {

  xt::pyarray<T> images_;
  size_t ma_window_ = 1;
  size_t new_ma_window_ = 1;
  size_t ma_count_;

public:
  explicit RawImageData(const xt::pyarray<T>& images)
    : images_(images), ma_count_(1) {
  };

  ~RawImageData() = default;

  // return the number of images per train
  size_t nImages() const {
    if (pulseResolved()) return images_.shape()[0];
    return 1;
  }

  bool pulseResolved() const { return images_.shape().size() == 3; }

  void setImages(const xt::pyarray<T>& images) {
    bool reset = false;
    if (new_ma_window_ < ma_window_) reset = true;
    ma_window_ = new_ma_window_;

    if (ma_window_ > 1 && images.shape() == images_.shape() && not reset) {
      if (ma_count_ < ma_window_) {
        ++ma_count_;
        images_ = images_ + (images - images_) / ma_count_;
      } else {
        images_ = images_ + (images - images_) / ma_window_;
      }
    } else {
      images_ = images;
      ma_count_ = 1;
    }
  };

  // Python code can modify the internal c++ data
  xt::pyarray<T>& getImages() { return images_; }

  void setMovingAverageWindow(size_t v) {
    if (! v) v = 1;
    new_ma_window_ = v;
  }

  size_t getMovingAverageWindow() const { return ma_window_; }

  size_t getMovingAverageCount() const { return ma_count_; }

  void clear() {
    ma_window_ = 1;
    ma_count_ = 0;
  }
};


namespace py = pybind11;

PYBIND11_MODULE(image_data, m) {
  xt::import_numpy();

  py::class_<RawImageData<float>>(m, "RawImageData")
  .def(py::init<const xt::pyarray<float>&>())
  .def("nImages", &RawImageData<float>::nImages)
  .def("pulseResolved", &RawImageData<float>::pulseResolved)
  .def("getImages", &RawImageData<float>::getImages)
  .def("setImages", &RawImageData<float>::setImages)
  .def("getMovingAverageWindow", &RawImageData<float>::getMovingAverageWindow)
  .def("setMovingAverageWindow", &RawImageData<float>::setMovingAverageWindow)
  .def("getMovingAverageCount", &RawImageData<float>::getMovingAverageCount);

}
