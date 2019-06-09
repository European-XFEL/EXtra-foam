pybind11_add_module(algo algo.cpp)

pybind11_add_module(xtnumpy xtnumpy.cpp)

target_include_directories(xtnumpy
    PRIVATE
        ${xtensor-python_INCLUDE_DIRS})

target_link_libraries(xtnumpy PRIVATE xtensor-python)