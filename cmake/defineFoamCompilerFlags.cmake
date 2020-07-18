###################################################################
# Author: Jun Zhu <jun.zhu@xfel.eu>                               #
# Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. #
# All rights reserved.                                            #
###################################################################

function(define_foam_compile_flags MODULE_NAME)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        if(NOT CMAKE_CXX_FLAGS MATCHES "-march")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        endif()
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")

    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

    if( ${MODULE_NAME} STREQUAL "foam_py" )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "7")
                # use '-faligned-new' to enable C++17 over-aligned new support
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
            endif()
        endif()

        message(STATUS "=============================== EXtra-foam Python =================================")
    elseif( ${MODULE_NAME} STREQUAL "foam_test" )
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "7")
                # solve the linker error with test_tbb.cpp:
                #     undefined reference to symbol 'clock_gettime@@GLIBC_2.2.5'
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt")
            endif()
        endif()

        message(STATUS "============================= EXtra-foam Unit Test ================================")
    endif()
    message(STATUS "")
    message(STATUS "    C++ Compiler:               ${FOAM_COMPILER_STR}")
    message(STATUS "    Build type:                 ${CMAKE_BUILD_TYPE}")
    message(STATUS "    C++ flags (Release):        ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "    C++ flags (Debug):          ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
    message(STATUS "    Linker flags (Release):     ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
    message(STATUS "    Linker flags (Debug):       ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
    message(STATUS "")
    message(STATUS "    USE TBB in FOAM/XTENSOR:    ${FOAM_USE_TBB}/${XTENSOR_USE_TBB}")
    message(STATUS "    USE XSIMD in FOAM/XTENSOR:  ${FOAM_USE_XSIMD}/${XTENSOR_USE_XSIMD}")
    message(STATUS "")
    message(STATUS "==================================================================================")

endfunction()
