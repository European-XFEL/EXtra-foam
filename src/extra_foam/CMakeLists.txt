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

# image_proc

pybind11_add_module(image_proc image_proc.cpp)

target_include_directories(image_proc
    PRIVATE
        ${FOAM_INCLUDE_DIRS}
        ${TBB_INCLUDE_DIRS})

target_link_libraries(image_proc
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
