pybind11_add_module(geometry geometry.cpp)
pybind11_add_module(pynumpy pynumpy.cpp)

target_include_directories(pynumpy
    PRIVATE
        ${PROJECT_SOURCE_DIR}/thirdparty/xtl/include
        ${PROJECT_SOURCE_DIR}/thirdparty/xtensor/include
)