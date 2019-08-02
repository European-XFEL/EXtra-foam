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

if(FAI_WITH_TBB)
    add_definitions(-DFAI_WITH_TBB)

    # TODO: try to find TBB in the system first?
    set(tbb_root_dir ${PROJECT_SOURCE_DIR}/thirdparty/tbb)

    include(${tbb_root_dir}/cmake/TBBBuild.cmake)
    tbb_build(TBB_ROOT ${PROJECT_SOURCE_DIR}/thirdparty/tbb
        CONFIG_DIR TBB_DIR
        MAKE_ARGS tbb_build_dir=${CMAKE_CURRENT_BINARY_DIR}/thirdparty/tbb
                  tbb_build_prefix=tbb_cmake_build)

    find_package(TBB REQUIRED tbb tbbmalloc)
    get_target_property(TBB_INCLUDE_DIRS TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(TBB_LIBRARIES TBB::tbb IMPORTED_LOCATION_RELEASE)

    message(STATUS "FOUND TBB: ${TBB_VERSION}, ${TBB_LIBRARIES}")

    set(XTENSOR_USE_TBB ON CACHE BOOL "Build XTENSOR with TBB")
endif()

# pybind11
add_subdirectory(thirdparty/pybind11)
set(pybind11_INCLUDE_DIRS PYBIND11_INCLUDE_DIR)

# xtl
add_subdirectory(thirdparty/xtl)
set(xtl_DIR ${PROJECT_BINARY_DIR}/thirdparty/xtl)

# xsimd
if(FAI_WITH_XSIMD)
    add_subdirectory(thirdparty/xsimd)
    set(xsimd_DIR ${PROJECT_BINARY_DIR}/thirdparty/xsimd)
    find_package(xsimd 7.2 REQUIRED)
    set(XTENSOR_USE_XSIMD ON CACHE BOOL "Build XTENSOR with XSIMD")
endif()

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
