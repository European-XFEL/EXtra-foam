#include <pybind11/pybind11.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xreducer.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor-python/pyarray.hpp"


template<typename T>
inline xt::pyarray<T> nanmeanImages(const xt::pyarray<T>& arr) {
  if (arr.shape().size() != 3)
    throw std::invalid_argument("Input must be a three dimensional array!");
  return xt::nanmean<T>(arr, {0}, xt::evaluation_strategy::immediate);
}


template<typename T>
inline T nanmeanScalar(T x, T y) {
  if (std::isnan(x) and std::isnan(y)) return 0;

  if (std::isnan(x)) return y;

  if (std::isnan(y)) return x;

  return (x + y) / 2;
}


namespace py = pybind11;


PYBIND11_MODULE(xtnumpy, m) {

  xt::import_numpy();

  m.doc() = "Calculate the mean of images, ignoring NaNs.";

  // only works for float since np.nan is a float?
  m.def("xt_nanmean_images", &nanmeanImages<double>);
  m.def("xt_nanmean_images", &nanmeanImages<float>);
  m.def("xt_nanmean_two_images", xt::pyvectorize(nanmeanScalar<double>));
  m.def("xt_nanmean_two_images", xt::pyvectorize(nanmeanScalar<float>));
}
