# algo

pybind11_add_module(algo algo.cpp)

# xtnumpy

pybind11_add_module(xtnumpy xtnumpy.cpp)

target_include_directories(xtnumpy
    PRIVATE
        ${xtensor-python_INCLUDE_DIRS})

target_link_libraries(xtnumpy PRIVATE xtensor-python)

# image data

pybind11_add_Module(image_data image_data.cpp)

target_include_directories(image_data
    PRIVATE
        ${xtensor-python_INCLUDE_DIRS})

target_link_libraries(image_data PRIVATE xtensor-python)

target_compile_options(image_data
    PRIVATE
        -fPIC)
