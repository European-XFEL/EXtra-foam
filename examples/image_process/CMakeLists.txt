cmake_minimum_required(VERSION 3.14)
project(foam_cpp_api_test)

set(CMAKE_CXX_STANDARD 14)

find_package(extra-foam REQUIRED)
if(extra-foam_FOUND)
    message(STATUS "Found EXtra-foam version: ${extra-foam_INCLUDE_DIR}")
endif()

set(TBB_INCLUDE_DIR ${CMAKE_PREFIX_PATH}/include)
set(TBB_LIBRARY ${CMAKE_PREFIX_PATH}/lib/libtbb.so)

add_executable(image_process main.cpp)
target_include_directories(image_process PRIVATE ${TBB_INCLUDE_DIR})
target_link_libraries(image_process PRIVATE extra-foam ${TBB_LIBRARY})
