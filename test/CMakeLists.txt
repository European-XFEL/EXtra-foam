###################################################################
# Author: Jun Zhu <jun.zhu@xfel.eu>                               #
# Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. #
# All rights reserved.                                            #
###################################################################

find_package(GTest REQUIRED)
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

include(defineFoamCompilerFlags)
define_foam_compile_flags(foam_test)

foreach(pathname IN LISTS _FOAM_UNITTESTS)
    get_filename_component(filename ${pathname} NAME)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} main.cpp ${filename})
    target_include_directories(${targetname}
        PRIVATE
            ${GTEST_INCLUDE_DIRS}
    )
    target_link_libraries(${targetname}
        PRIVATE
            extra-foam
            $<$<OR:$<BOOL:${FOAM_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:TBB::tbb>
            GTest::gtest
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
        ${GTEST_INCLUDE_DIRS})

target_link_libraries(test_foam_cpp
    PRIVATE
        GTest::gtest
        pthread
        extra-foam
        $<$<OR:$<BOOL:${FOAM_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:TBB::tbb>
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES})

add_custom_target(ftest
    COMMAND test_foam_cpp
    DEPENDS test_foam_cpp
)
