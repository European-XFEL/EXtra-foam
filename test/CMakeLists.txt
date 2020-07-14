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

set(_FOAM_UNITTESTS
    test_tbb.cpp
    test_azimuthal_integrator.cpp
    test_blas.cpp
    test_canny.cpp
    test_geometry.cpp
    test_geometry_1m.cpp
    test_imageproc.cpp
    test_smooth.cpp
    test_statistics.cpp
    test_traits.cpp
    )

set(_FOAM_UNITTEST_TARGETS_NEED_BLAS
    test_blas)

set(CMAKE_BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    if(NOT CMAKE_CXX_FLAGS MATCHES "-march")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "7")
        # solve the linker error with test_tbb.cpp:
        #     undefined reference to symbol 'clock_gettime@@GLIBC_2.2.5'
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

foreach(pathname IN LISTS _FOAM_UNITTESTS)
    get_filename_component(filename ${pathname} NAME)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} main.cpp ${filename})
    target_include_directories(${targetname}
        PRIVATE
            $<$<OR:$<BOOL:${FOAM_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_INCLUDE_DIRS}>
            ${GTEST_INCLUDE_DIRS}
            ${GMOCK_INCLUDE_DIRS}
    )
    target_link_libraries(${targetname}
        PRIVATE
            extra-foam
            $<$<OR:$<BOOL:${FOAM_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_LIBRARY}>
            gtest
            pthread)

    if (${targetname} IN_LIST _FOAM_UNITTEST_TARGETS_NEED_BLAS)
        target_link_libraries(${targetname}
            PRIVATE
                ${BLAS_LIBRARIES}
                ${LAPACK_LIBRARIES})
    endif()

    add_custom_target(f${targetname}
        COMMAND ${targetname}
        DEPENDS ${targetname} ${filename}
    )
endforeach()

add_executable(test_foam_cpp main.cpp ${_FOAM_UNITTESTS})

target_include_directories(test_foam_cpp
    PRIVATE
        $<$<OR:$<BOOL:${FOAM_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_INCLUDE_DIRS}>
        ${GTEST_INCLUDE_DIRS}
        ${GMOCK_INCLUDE_DIRS})

target_link_libraries(test_foam_cpp
    PRIVATE
        gtest
        pthread
        extra-foam
        $<$<OR:$<BOOL:${FOAM_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_LIBRARY}>
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES})

add_custom_target(ftest
    COMMAND test_foam_cpp
    DEPENDS test_foam_cpp
)
