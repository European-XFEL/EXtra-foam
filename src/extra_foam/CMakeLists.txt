###################################################################
# Author: Jun Zhu <jun.zhu@xfel.eu>                               #
# Copyright (C) European X-Ray Free-Electron Laser Facility GmbH. #
# All rights reserved.                                            #
###################################################################

set(extra-foam_MODULE_FILES
    f_helpers.cpp
    f_imageproc.cpp
    f_datamodel.cpp
    f_geometry.cpp
    f_geometry_1m.cpp
    f_statistics.cpp
    f_smooth.cpp
    f_canny.cpp
)

set(FOAM_MODULE_WITH_BLAS)

set(CMAKE_INSTALL_RPATH "${target_install_rpath};${CMAKE_PREFIX_PATH}/lib")

foreach(filename IN LISTS extra-foam_MODULE_FILES)
    string(REPLACE ".cpp" "" modulename ${filename})
    string(REGEX REPLACE "^f_" "" modulename ${modulename})
    pybind11_add_module(${modulename} ${filename})
    target_include_directories(${modulename} PRIVATE include)
    target_link_libraries(${modulename} PRIVATE xtensor-python)

    if(FOAM_WITH_TBB OR XTENSOR_USE_TBB)
        target_include_directories(${modulename} PRIVATE ${TBB_INCLUDE_DIRS})
        target_link_libraries(${modulename} PRIVATE ${TBB_LIBRARY})
    endif()

    if (${modulename} IN_LIST FOAM_MODULE_WITH_BLAS)
        target_link_libraries(${modulename}
            PRIVATE
                xtensor-blas
                ${BLAS_LIBRARIES}
                ${LAPACK_LIBRARIES})
    endif()
endforeach()
