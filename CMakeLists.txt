###################################################################
# Author: Jun Zhu <jun.zhu@xfel.eu>                               #
# Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. #
# All rights reserved.                                            #
###################################################################

cmake_minimum_required(VERSION 3.8)

if(${CMAKE_VERSION} VERSION_LESS 3.13)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
    cmake_policy(VERSION 3.13)
endif()

project(karaboFAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

OPTION(FAI_WITH_TBB "Build karaboFAI (xtensor is not included) with intel TBB" ON)

OPTION(FAI_WITH_XSIMD "Build karaboFAI (xtensor is not included) with XSIMD" ON)

OPTION(BUILD_FAI_TESTS "Build c++ unit test" OFF)

# tbb
if(FAI_WITH_TBB OR XTENSOR_USE_TBB)
    set(tbb_root_dir ${PROJECT_SOURCE_DIR}/thirdparty/tbb)

    include(${tbb_root_dir}/cmake/TBBBuild.cmake)
    tbb_build(TBB_ROOT ${PROJECT_SOURCE_DIR}/thirdparty/tbb
        CONFIG_DIR TBB_DIR
        MAKE_ARGS tbb_build_dir=${CMAKE_CURRENT_BINARY_DIR}/thirdparty/tbb
                  tbb_build_prefix=tbb_cmake_build)

    find_package(TBB REQUIRED tbb tbbmalloc)
    # TBB_FOUND will silence xtensor/cmake/FindTBB.cmake, which does not work here
    get_target_property(TBB_INCLUDE_DIRS TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(TBB_LIBRARIES TBB::tbb IMPORTED_LOCATION_RELEASE)

    message(STATUS "Found TBB: ${TBB_VERSION}, ${TBB_LIBRARIES}")

    if(FAI_WITH_TBB)
        message(STATUS "Build karaboFAI with intel TBB")

        if(${CMAKE_VERSION} VERSION_LESS 3.13)
            add_definitions(-DFAI_WITH_TBB)
        else()
            add_compile_definitions(FAI_WITH_TBB)
        endif()
    endif()
endif()

# xsimd
if(FAI_WITH_XSIMD OR XTENSOR_USE_XSIMD)
    add_subdirectory(thirdparty/xsimd)
    set(xsimd_DIR ${PROJECT_BINARY_DIR}/thirdparty/xsimd)

    if(FAI_WITH_XSIMD)
        message(STATUS "Build karaboFAI with XSIMD")

        if(${CMAKE_VERSION} VERSION_LESS 3.13)
            add_definitions(-DFAI_WITH_XSIMD)
        else()
            add_compile_definitions(FAI_WITH_XSIMD)
        endif()
    endif()

endif()

# pybind11
add_subdirectory(thirdparty/pybind11)
find_package(pybind11 REQUIRED)

# xtl
add_subdirectory(thirdparty/xtl)
set(xtl_DIR ${CMAKE_BINARY_DIR}/thirdparty/xtl)

# xtensor
add_subdirectory(thirdparty/xtensor)
set(xtensor_DIR ${CMAKE_BINARY_DIR}/thirdparty/xtensor)

# xtensor-python
add_subdirectory(thirdparty/xtensor-python)

# karaboFAI
add_subdirectory(src/karaboFAI)

if(BUILD_FAI_TESTS)
    add_subdirectory(test)
endif()
