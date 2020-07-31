###################################################################
# Author: Jun Zhu <jun.zhu@xfel.eu>                               #
# Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. #
# All rights reserved.                                            #
###################################################################

cmake_minimum_required(VERSION 3.13)
project(extra-foam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
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
if(FOAM_USE_TBB OR XTENSOR_USE_TBB)
    set(TBB_REQUIRED_VERSION 2020.1)
    string(REPLACE "." "_U" TBB_REQUIRED_VERSION_STR ${TBB_REQUIRED_VERSION})

    add_compile_definitions(TBB_SUPPRESS_DEPRECATED_MESSAGES)

    find_package(TBB QUIET)
    if(TBB_FOUND AND ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} VERSION_EQUAL TBB_REQUIRED_VERSION)
        set(TBB_LIBRARY ${TBB_LIBRARIES_RELEASE})
    else()
        message(STATUS "Downloading TBB ${TBB_REQUIRED_VERSION_STR}")

        configure_file(cmake/downloadTbb.cmake.in ${thirdparty_BINARY_DIR}/tbb-stage/CMakeLists.txt)
        setup_external_project( tbb )

        include(${thirdparty_BINARY_DIR}/tbb-src/cmake/TBBBuild.cmake)
        tbb_build(TBB_ROOT ${thirdparty_BINARY_DIR}/tbb-src
                  CONFIG_DIR TBB_DIR
                  MAKE_ARGS tbb_build_dir=${thirdparty_BINARY_DIR}/tbb-build
                            tbb_build_prefix=tbb_cmake_build)
        find_package(TBB ${TBB_REQUIRED_VERSION} EXACT CONFIG REQUIRED COMPONENTS tbb tbbmalloc)

        get_target_property(TBB_INCLUDE_DIRS TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(TBB_LIBRARY TBB::tbb IMPORTED_LOCATION_RELEASE)
    endif()

    message(STATUS "TBB FOUND ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}: ${TBB_INCLUDE_DIRS}, ${TBB_LIBRARY}")

    if(FOAM_USE_TBB)
        message(STATUS "Build extra-foam with intel TBB")
        add_compile_definitions(FOAM_USE_TBB)
    endif()
endif()

# xsimd
if(FOAM_USE_XSIMD OR XTENSOR_USE_XSIMD)
    add_subdirectory(thirdparty/xsimd)
    set(xsimd_DIR ${PROJECT_BINARY_DIR}/thirdparty/xsimd)

    if(FOAM_USE_XSIMD)
        message(STATUS "Build extra-foam with XSIMD")
        add_compile_definitions(FOAM_USE_XSIMD)
    endif()

endif()

# xtl
add_subdirectory(thirdparty/xtl)
set(xtl_DIR ${CMAKE_BINARY_DIR}/thirdparty/xtl)

# xtensor
add_subdirectory(thirdparty/xtensor)
set(xtensor_DIR ${CMAKE_BINARY_DIR}/thirdparty/xtensor)

# xtensor-blas
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})
add_subdirectory(thirdparty/xtensor-blas)

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
    # pybind11
    set(PYBIND11_CPP_STANDARD -std=c++14)
    add_subdirectory(thirdparty/pybind11)
    find_package(pybind11 REQUIRED)

    # xtensor-python
    add_subdirectory(thirdparty/xtensor-python)

    # extra-foam Python
    add_subdirectory(src/extra_foam)
endif()
