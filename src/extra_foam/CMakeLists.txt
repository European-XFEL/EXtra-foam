###################################################################
# Author: Jun Zhu <jun.zhu@xfel.eu>                               #
# Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. #
# All rights reserved.                                            #
###################################################################

find_package(pybind11 REQUIRED CONFIG)
find_package(xtensor-python REQUIRED)

set(Python3_FIND_VIRTUALENV "ONLY")
find_package(Python3 REQUIRED COMPONENTS NumPy)

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
            ${Python3_NumPy_INCLUDE_DIRS})
    target_link_libraries(${modulename}
        PRIVATE
            extra-foam
            $<$<OR:$<BOOL:${FOAM_USE_TBB}>,$<BOOL:${XTENSOR_USE_TBB}>>:TBB::tbb>
            xtensor-python)
endforeach()
