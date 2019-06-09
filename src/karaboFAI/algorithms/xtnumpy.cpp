#include <pybind11/pybind11.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xreducer.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"


template<typename T>
xt::pyarray<T> xt_nanmean_image(const xt::pyarray<T>& arr) {
  if (arr.shape().size() != 3)
    throw std::invalid_argument("Input must be a three dimensional array!");
  return xt::nanmean<T>(arr, {0}, xt::evaluation_strategy::immediate);
}


namespace py = pybind11;


PYBIND11_MODULE(xtnumpy, m) {

  xt::import_numpy();

  m.doc() = "Calculate the mean of images, ignoring NaNs.";

  // only works for float since np.nan is a float?
  m.def("xt_nanmean_image", &xt_nanmean_image<double>);
  m.def("xt_nanmean_image", &xt_nanmean_image<float>);
}
