cmake_minimum_required(VERSION 3.13)
project(live_view)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC_ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# karabo bridge
find_package(karabo-bridge REQUIRED CONFIG)
message(STATUS "Found karabo-bridge ${karabo-bridge_INCLUDE_DIRS}")

# qt
find_package(Qt5Core REQUIRED)
message(STATUS "Found Qt${Qt5Core_VERSION}: ${Qt5Core_INCLUDE_DIRS}")
find_package(Qt5Gui REQUIRED)
message(STATUS "Found Qt${Qt5Gui_VERSION}: ${Qt5Gui_INCLUDE_DIRS}")
find_package(Qt5Widgets REQUIRED)
message(STATUS "Found Qt${Qt5Widgets_VERSION}: ${Qt5Widgets_INCLUDE_DIRS}")

# extra-foam
find_package(extra-foam REQUIRED)
if(extra-foam_FOUND)
    message(STATUS "Found EXtra-foam version: ${extra-foam_INCLUDE_DIR}")
endif()

set(TBB_INCLUDE_DIR ${CMAKE_PREFIX_PATH}/include)
set(TBB_LIBRARY ${CMAKE_PREFIX_PATH}/lib/libtbb.so)

add_executable(${PROJECT_NAME}
    main.cpp
    image_view.cpp
    image_view.hpp
    broker.cpp
    broker.hpp
    image_processor.cpp
    image_processor.hpp
    config.hpp
    )

target_include_directories(${PROJECT_NAME} PRIVATE ${TBB_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        extra-foam
        karabo-bridge
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        ${TBB_LIBRARY})

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DQT_NO_DEBUG_OUTPUT)
endif()
