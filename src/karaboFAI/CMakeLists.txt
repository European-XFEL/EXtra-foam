# algo

pybind11_add_module(algo algo.cpp)

# xtnumpy

pybind11_add_module(xtnumpy xtnumpy.cpp)

target_include_directories(xtnumpy
    PRIVATE
        ${xtensor-python_INCLUDE_DIRS})

target_link_libraries(xtnumpy PRIVATE xtensor-python)

# image data

pybind11_add_Module(data_model data_model.cpp)

target_include_directories(data_model
    PRIVATE
        ${xtensor-python_INCLUDE_DIRS})

target_link_libraries(data_model PRIVATE xtensor-python)

target_compile_options(data_model
    PRIVATE
        -fPIC)
