###################################################################
# Author: Jun Zhu <jun.zhu@xfel.eu>                               #
# Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. #
# All rights reserved.                                            #
###################################################################

cmake_minimum_required(VERSION 3.17)
project(extra-foam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# options
# =======

OPTION(FOAM_USE_TBB "Build extra-foam (xtensor is not included) with intel TBB" OFF)

OPTION(FOAM_USE_XSIMD "Build extra-foam (xtensor is not included) with XSIMD" OFF)

OPTION(BUILD_FOAM_TESTS "Build c++ unit test" OFF)

OPTION(BUILD_FOAM_PYTHON "Build FOAM Python bindings" OFF)

# dependencies
# ============

# tbb
if(FOAM_USE_TBB OR XTENSOR_USE_TBB)
    find_package(TBB 2021 REQUIRED)
    message(STATUS "TBB FOUND ${TBB_VERSION}")

    if(FOAM_USE_TBB)
        message(STATUS "Build extra-foam with intel TBB")
        add_compile_definitions(FOAM_USE_TBB)
    endif()
endif()

# xsimd
if(FOAM_USE_XSIMD OR XTENSOR_USE_XSIMD)
    find_package(xsimd)

    if(FOAM_USE_XSIMD)
        message(STATUS "Build extra-foam with XSIMD")
        add_compile_definitions(FOAM_USE_XSIMD)
    endif()

endif()

# xtensor
find_package(xtensor REQUIRED)

# xtensor-blas
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})
find_package(xtensor-blas)

# EXtra-foam
set(FOAM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/extra_foam/include)

set(FOAM_HEADERS
    ${FOAM_INCLUDE_DIR}/f_canny.hpp
    ${FOAM_INCLUDE_DIR}/f_geometry.hpp
    ${FOAM_INCLUDE_DIR}/f_geometry_1m.hpp
    ${FOAM_INCLUDE_DIR}/f_helpers.hpp
    ${FOAM_INCLUDE_DIR}/f_imageproc.hpp
    ${FOAM_INCLUDE_DIR}/f_smooth.hpp
    ${FOAM_INCLUDE_DIR}/f_statistics.hpp
    ${FOAM_INCLUDE_DIR}/f_traits.hpp
    ${FOAM_INCLUDE_DIR}/f_utilities.hpp
    )

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${FOAM_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME}
    INTERFACE
        xtensor
        xtensor-blas
        $<$<BOOL:${FOAM_USE_XSIMD}>:xsimd>)

# ============
# Build flags
# ============

if(CMAKE_CXX_COMPILER_VERSION)
    set(FOAM_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (version ${CMAKE_CXX_COMPILER_VERSION})")
else()
    set(FOAM_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}")
endif()
string(STRIP "${FOAM_COMPILER_STR}" FOAM_COMPILER_STR)

# ============
# Unit test
# ============

if(BUILD_FOAM_TESTS)
    add_subdirectory(test)
endif()

# ============
# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(FILES ${FOAM_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

configure_package_config_file(
    ${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_NAME}_VERSION
    COMPATIBILITY AnyNewerVersion
)

install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# ============
# FOAM_PYTHON
# ============

if(BUILD_FOAM_PYTHON)
    # extra-foam Python
    add_subdirectory(src/extra_foam)
endif()
