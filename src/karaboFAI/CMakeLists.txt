###################################################################
# Author: Jun Zhu <jun.zhu@xfel.eu>                               #
# Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. #
# All rights reserved.                                            #
###################################################################

# TODO: separate xtensor-python binding with others in the fugure
set(xtensor-python_DIR ${CMAKE_BINARY_DIR}/thirdparty/xtensor-python)
find_package(xtensor-python)


# algo

pybind11_add_module(algo algo.cpp)

# xtnumpy

pybind11_add_module(xtnumpy xtnumpy.cpp)

target_include_directories(xtnumpy
    PRIVATE
        ${FAI_INCLUDE_DIRS}
        ${TBB_INCLUDE_DIRS})

target_link_libraries(xtnumpy
    PRIVATE
        xtensor-python
        ${TBB_LIBRARIES})

# image data

pybind11_add_Module(data_model data_model.cpp)

target_include_directories(data_model
    PRIVATE
        ${xtensor-python_INCLUDE_DIRS})

target_link_libraries(data_model PRIVATE xtensor-python)

target_compile_options(data_model
    PRIVATE
        -fPIC)
