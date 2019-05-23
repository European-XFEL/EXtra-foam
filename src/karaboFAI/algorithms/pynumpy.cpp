#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>

#include "xtensor/xarray.hpp"


double first_tensor() {
  xt::xarray<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

  return arr1(0, 0);
}

namespace py = pybind11;


PYBIND11_MODULE(pynumpy, m) {
  m.def("first_tensor", &first_tensor);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}