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
)

if(UNIX)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

if(APPLE)
  set(target_install_rpath "@loader_path/")
else()
  set(target_install_rpath "\$ORIGIN")
endif()

foreach(filename IN LISTS extra-foam_MODULE_FILES)
    string(REPLACE ".cpp" "" modulename ${filename})
    string(REGEX REPLACE "^f_" "" modulename ${modulename})
    # set(modulename _${modulename})
    pybind11_add_module(${modulename} ${filename})
    target_include_directories(${modulename} PRIVATE include)
    target_link_libraries(${modulename} PRIVATE xtensor-python)
    set_target_properties(${modulename} PROPERTIES INSTALL_RPATH ${target_install_rpath})

    if(FOAM_WITH_TBB OR XTENSOR_USE_TBB)
        target_include_directories(${modulename} PRIVATE ${TBB_INCLUDE_DIRS})
        target_link_libraries(${modulename} PRIVATE ${TBB_LIBRARIES})
    endif()
endforeach()
