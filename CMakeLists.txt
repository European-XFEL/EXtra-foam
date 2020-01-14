###################################################################
# Author: Jun Zhu <jun.zhu@xfel.eu>                               #
# Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. #
# All rights reserved.                                            #
###################################################################

cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_LESS 3.13)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
    cmake_policy(VERSION 3.13)
endif()

project(extra-foam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

OPTION(FOAM_WITH_TBB "Build extra-foam (xtensor is not included) with intel TBB" ON)

OPTION(FOAM_WITH_XSIMD "Build extra-foam (xtensor is not included) with XSIMD" ON)

OPTION(BUILD_FOAM_TESTS "Build c++ unit test" OFF)

set(thirdparty_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty)

function(setup_external_project NAME)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${thirdparty_BINARY_DIR}/${NAME}-stage
    )
    if(result)
        message(FATAL_ERROR "CMake step for ${NAME} failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${thirdparty_BINARY_DIR}/${NAME}-stage
    )
    if(result)
        message(FATAL_ERROR "Build step for ${NAME} failed: ${result}")
    endif()
endfunction()

# tbb
if(FOAM_WITH_TBB OR XTENSOR_USE_TBB)
    set(tbb_REQUIRED_VERSION 2019_U9)

    configure_file(cmake/downloadTbb.cmake.in ${thirdparty_BINARY_DIR}/tbb-stage/CMakeLists.txt)
    setup_external_project( tbb )

    include(${thirdparty_BINARY_DIR}/tbb-src/cmake/TBBBuild.cmake)
    tbb_build(TBB_ROOT ${thirdparty_BINARY_DIR}/tbb-src
              CONFIG_DIR TBB_DIR
              MAKE_ARGS tbb_build_dir=${thirdparty_BINARY_DIR}/tbb-build
                        tbb_build_prefix=tbb_cmake_build)
    find_package(TBB CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
    message(STATUS "Found TBB: ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}")

    get_target_property(TBB_INCLUDE_DIRS TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(TBB_LIBRARIES TBB::tbb IMPORTED_LOCATION_RELEASE)

    get_filename_component(TBB_LIB_NAME ${TBB_LIBRARIES} NAME)
    configure_file(${TBB_LIBRARIES} ${CMAKE_CURRENT_BINARY_DIR}/${TBB_LIB_NAME} COPYONLY)

    if(FOAM_WITH_TBB)
        message(STATUS "Build extra-foam with intel TBB")
        add_compile_definitions(FOAM_WITH_TBB)
    endif()
endif()

# xsimd
if(FOAM_WITH_XSIMD OR XTENSOR_USE_XSIMD)
    add_subdirectory(thirdparty/xsimd)
    set(xsimd_DIR ${PROJECT_BINARY_DIR}/thirdparty/xsimd)

    if(FOAM_WITH_XSIMD)
        message(STATUS "Build extra-foam with XSIMD")
        add_compile_definitions(FOAM_WITH_XSIMD)
    endif()

endif()

# pybind11
set(PYBIND11_CPP_STANDARD -std=c++14)
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

# extra-foam
set(FOAM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src/extra_foam/include)

add_subdirectory(src/extra_foam)

if(BUILD_FOAM_TESTS)
    add_subdirectory(test)
endif()
