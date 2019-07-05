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

# pybind11
add_subdirectory(thirdparty/pybind11)
set(pybind11_INCLUDE_DIRS PYBIND11_INCLUDE_DIR)

# xtl
add_subdirectory(thirdparty/xtl)
set(xtl_DIR ${PROJECT_BINARY_DIR}/thirdparty/xtl)

# xsimd
add_subdirectory(thirdparty/xsimd)
set(xsimd_DIR ${PROJECT_BINARY_DIR}/thirdparty/xsimd)

# xtensor
add_subdirectory(thirdparty/xtensor)
set(xtensor_DIR ${PROJECT_BINARY_DIR}/thirdparty/xtensor)
set(xtensor_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/xtensor/include)

# xtensor-python
set(xtensor-python_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/xtensor-python/include)

find_package(NumPy REQUIRED)

add_library(xtensor-python INTERFACE)
target_include_directories(xtensor-python INTERFACE
    "$<BUILD_INTERFACE:${xtensor-python_INCLUDE_DIR}>;${NUMPY_INCLUDE_DIRS}"
    $<INSTALL_INTERFACE:include>)
target_link_libraries(xtensor-python INTERFACE xtensor)

add_subdirectory(src/karaboFAI)

OPTION(BUILD_FAI_TESTS "test suit" OFF)

if(BUILD_FAI_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
