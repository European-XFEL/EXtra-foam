###################################################################
# Author: Jun Zhu <jun.zhu@xfel.eu>                               #
# Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. #
# All rights reserved.                                            #
###################################################################

# Download and unpack googletest at configure time
configure_file(downloadGTest.cmake.in googletest-download/CMakeLists.txt)

execute_process(
    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
)
if (result)
    message(FATAL_ERROR "CMAKE step for gtest failed: ${result}")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
)
if (result)
    message(FATAL_ERROR "BUILD step for gtest failed: ${result}")
endif()

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds the following targets:
# gtest, gtest_main, gmock and gmock_main
add_subdirectory(
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-src
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-build EXCLUDE_FROM_ALL
)

set(GTEST_INCLUDE_DIRS "${gtest_SOURCE_DIR}/include")
set(GMOCK_INCLUDE_DIRS "${gmock_SOURCE_DIR}/include")

find_package(Threads REQUIRED)

set(FAI_TESTS
    test_tbb.cpp
    test_image_proc.cpp
)

foreach(filename IN LISTS FAI_TESTS)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} main.cpp ${filename})
    if(FAI_WITH_TBB)
        target_compile_definitions(${targetname} PRIVATE FAI_WITH_TBB)
        target_include_directories(${targetname} PRIVATE ${TBB_INCLUDE_DIRS})
        target_link_libraries(${targetname} PRIVATE ${TBB_LIBRARIES})
    endif()
    target_include_directories(${targetname}
        PRIVATE
            ${FAI_INCLUDE_DIRS}
            ${GTEST_INCLUDE_DIRS}
            ${GMOCK_INCLUDE_DIRS}
    )
    target_link_libraries(${targetname} PRIVATE gtest pthread xtensor)
    add_custom_target(
        f${targetname}
        COMMAND ${targetname}
        DEPENDS ${targetname} ${filename}
    )
endforeach()

add_executable(test_fai_cpp main.cpp ${FAI_TESTS})

target_include_directories(test_fai_cpp
    PRIVATE
        ${FAI_INCLUDE_DIRS}
        ${GTEST_INCLUDE_DIRS}
        ${GMOCK_INCLUDE_DIRS}
)

target_link_libraries(test_fai_cpp PRIVATE gtest pthread xtensor)

if(FAI_WITH_TBB)
    target_compile_definitions(test_fai_cpp PRIVATE FAI_WITH_TBB)
    target_include_directories(test_fai_cpp PRIVATE ${TBB_INCLUDE_DIRS})
    target_link_libraries(test_fai_cpp PRIVATE ${TBB_LIBRARIES})
endif()

add_custom_target(
    ftest
    COMMAND test_fai_cpp
    DEPENDS test_fai_cpp
)
