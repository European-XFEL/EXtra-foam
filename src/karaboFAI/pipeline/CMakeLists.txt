pybind11_add_Module(image_data image_data.cpp)

target_include_directories(image_data
    PRIVATE
        ${xtensor-python_INCLUDE_DIRS})

target_link_libraries(image_data PRIVATE xtensor-python)

target_compile_options(image_data
    PRIVATE
        -fPIC
    )
