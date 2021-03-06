###################################################################
# Author: Jun Zhu <jun.zhu@xfel.eu>                               #
# Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. #
# All rights reserved.                                            #
###################################################################

include(defineFoamCompilerFlags)
define_foam_compile_flags(foam_py)

# ===============
# Build libraries
# ===============

set(_FOAM_MODULE_FILES
    f_azimuthal_integrator.cpp
    f_canny.cpp
    f_datamodel.cpp
    f_geometry.cpp
    f_geometry_1m.cpp
    f_helpers.cpp
    f_imageproc.cpp
    f_smooth.cpp
    f_statistics.cpp
)

set(_FOAM_MODULES_NEED_BLAS)

if(APPLE)
    set(target_install_rpath "@loader_path/")
else()
    set(target_install_rpath "\$ORIGIN")
endif()
set(CMAKE_INSTALL_RPATH "${target_install_rpath};${CMAKE_PREFIX_PATH}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

foreach(pathname IN LISTS _FOAM_MODULE_FILES)
    get_filename_component(filename ${pathname} NAME)
    string(REPLACE ".cpp" "" modulename ${filename})
    string(REGEX REPLACE "^f_" "" modulename ${modulename})
    pybind11_add_module(${modulename} ${filename})
    target_include_directories(${modulename}
        PRIVATE
            include
            $<$<OR:$<BOOL:${FOAM_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_INCLUDE_DIRS}>)
    target_link_libraries(${modulename}
        PRIVATE
            extra-foam
            $<$<OR:$<BOOL:${FOAM_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:${TBB_LIBRARY}>
            xtensor-python)

    if (${modulename} IN_LIST _FOAM_MODULES_NEED_BLAS)
        target_link_libraries(${modulename} PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()
endforeach()
